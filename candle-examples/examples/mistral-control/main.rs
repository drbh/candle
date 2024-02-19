#![allow(non_snake_case, dead_code, unused_mut)]

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::Parser;

mod mistral;

use mistral::Config;
use mistral::{DecoderLayer, RmsNorm, RotaryEmbedding};

use candle_transformers::models::with_tracing::{linear_no_bias, Linear};

use candle::IndexOp;
use candle::{DType, Device, Module, Tensor, D};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::sync::Arc;
use tokenizers::Tokenizer;

enum Model {
    Mistral(Mistral),
}

#[derive(Debug, Clone)]
pub struct Mistral {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    sliding_window: usize,
    device: Device,
    dtype: DType,
}

#[derive(Debug, Clone)]
pub enum ControlVector<'a> {
    CaptureVec(&'a Vec<Vec<f32>>),
    Capture(Tensor),
    Apply(Tensor),
}

impl Mistral {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("model");
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let rotary_emb = Arc::new(RotaryEmbedding::new(vb.dtype(), cfg, vb_m.device())?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = DecoderLayer::new(rotary_emb.clone(), cfg, vb_l.pp(layer_idx))?;
            layers.push(layer)
        }
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;
        let lm_head = linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            sliding_window: cfg.sliding_window,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    fn prepare_decoder_attention_mask(
        &self,
        b_size: usize,
        tgt_len: usize,
        seqlen_offset: usize,
    ) -> candle::Result<Tensor> {
        // Sliding window mask?
        let mask: Vec<_> = (0..tgt_len)
            .flat_map(|i| {
                (0..tgt_len).map(move |j| {
                    if i < j || j + self.sliding_window < i {
                        f32::NEG_INFINITY
                    } else {
                        0.
                    }
                })
            })
            .collect();
        let mask = Tensor::from_slice(&mask, (tgt_len, tgt_len), &self.device)?;
        let mask = if seqlen_offset > 0 {
            let mask0 = Tensor::zeros((tgt_len, seqlen_offset), DType::F32, &self.device)?;
            Tensor::cat(&[&mask0, &mask], D::Minus1)?
        } else {
            mask
        };
        mask.expand((b_size, 1, tgt_len, tgt_len + seqlen_offset))?
            .to_dtype(self.dtype)
    }

    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        control_vector: &mut Option<&mut Vec<Vec<f32>>>,
        apply_control_vector: bool,
        last_hidden_states: &mut Tensor,
    ) -> candle::Result<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;
        let attention_mask = if seq_len <= 1 {
            None
        } else {
            let mask = self.prepare_decoder_attention_mask(b_size, seq_len, seqlen_offset)?;
            Some(mask)
        };
        let mut xs = self.embed_tokens.forward(input_ids)?;

        let len = self.layers.len();

        for (index, layer) in self.layers.iter_mut().enumerate() {
            xs = layer.forward(&xs, attention_mask.as_ref(), seqlen_offset)?;

            // apply layer norm to the last layer inside of the loop
            if index == len - 1 {
                xs = xs.apply(&self.norm)?;
            }

            // apply control vector after forward pass (and layer norm if last layer)
            if let Some(control_vector) = control_vector.as_mut() {
                if apply_control_vector {
                    let n = 14;
                    let m = 27;
                    if index < n || index > m {
                        // only apply control vector to intermediate layers
                        // skip the first N layers and the last M layers
                        // based on non scientific observation it seems like applying
                        // control vectors early can cause the model to diverge and applying
                        // control vectors to the last layers doesn't seem to have as much effect
                        // so applying to the middle layers is most stable
                    } else {
                        // println!("Applying control vector to layer {}", index);
                        if xs.dims()[1] == 1 {
                            let cv = &mut control_vector[index];
                            let cv = Tensor::from_slice(cv, (1, 1, 4096), &self.device)?
                                .to_dtype(self.dtype)?;
                            xs = (xs + cv)?;
                        }
                    }
                }
            }

            let last_item = xs.narrow(1, seq_len - 1, 1)?;
            // update last hidden states after forward pass
            *last_hidden_states =
                last_hidden_states.slice_assign(&[0..1, index..index + 1, 0..4096], &last_item)?;
        }
        xs.narrow(1, seq_len - 1, 1)?.apply(&self.lm_head)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache()
        }
    }
}

struct TextGeneration {
    model: Model,
    device: Device,
    tokenizer: TokenOutputStream,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
    control_vector: Vec<Vec<f32>>,
    apply_control_vector: bool,
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: Model,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        control_vector: Vec<Vec<f32>>,
        apply_control_vector: bool,
        device: &Device,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);
        Self {
            model,
            tokenizer: TokenOutputStream::new(tokenizer),
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            control_vector,
            apply_control_vector,
            device: device.clone(),
        }
    }

    fn run(
        &mut self,
        last_hidden_states: &mut Tensor,
        prompt: Vec<String>,
        sample_len: usize,
    ) -> Result<()> {
        use std::io::Write;
        self.tokenizer.clear();

        for (i, prompt) in prompt.iter().enumerate() {
            let start = std::time::Instant::now();

            let mut tokens = self
                .tokenizer
                .tokenizer()
                .encode(prompt.to_string(), true)
                .map_err(E::msg)?
                .get_ids()
                .to_vec();
            for &t in tokens.iter() {
                if let Some(t) = self.tokenizer.next_token(t)? {
                    print!("{t}")
                }
            }
            std::io::stdout().flush()?;

            let mut generated_tokens = 0usize;
            let eos_token = match self.tokenizer.get_token("</s>") {
                Some(token) => token,
                None => anyhow::bail!("cannot find the </s> token"),
            };
            let start_gen = std::time::Instant::now();
            for index in 0..sample_len {
                let context_size = if index > 0 { 1 } else { tokens.len() };
                let start_pos = tokens.len().saturating_sub(context_size);
                let ctxt = &tokens[start_pos..];
                let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
                let mut layers_last_hidden_states =
                    Tensor::zeros((1, 32, 4096), DType::BF16, &self.device)?;
                let logits = match &mut self.model {
                    Model::Mistral(m) => {
                        m.forward(
                            &input, //
                            start_pos,
                            &mut Some(&mut self.control_vector),
                            self.apply_control_vector,
                            &mut layers_last_hidden_states,
                        )?
                    }
                };

                // update last hidden states after forward pass
                *last_hidden_states = last_hidden_states
                    .slice_assign(&[i..i + 1, 0..32, 0..4096], &layers_last_hidden_states)?;

                let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
                let logits = if self.repeat_penalty == 1. {
                    logits
                } else {
                    let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                    candle_transformers::utils::apply_repeat_penalty(
                        &logits,
                        self.repeat_penalty,
                        &tokens[start_at..],
                    )?
                };

                let next_token = self.logits_processor.sample(&logits)?;
                tokens.push(next_token);
                generated_tokens += 1;
                if next_token == eos_token {
                    break;
                }
                if let Some(t) = self.tokenizer.next_token(next_token)? {
                    print!("{t}");
                    std::io::stdout().flush()?;
                }
            }

            let dt = start_gen.elapsed();
            if let Some(rest) = self.tokenizer.decode_rest().map_err(E::msg)? {
                print!("{rest}");
            }
            std::io::stdout().flush()?;
            println!(
                "\n{generated_tokens} tokens generated ({:.2} token/s)",
                generated_tokens as f64 / dt.as_secs_f64(),
            );

            match &mut self.model {
                Model::Mistral(m) => m.clear_kv_cache(),
            }
            println!("generated in {:?}", start.elapsed());
        }

        Ok(())
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    #[arg(long)]
    use_flash_attn: bool,

    #[arg(long)]
    prompt: Option<String>,

    /// The temperature used to generate samples.
    #[arg(long)]
    temperature: Option<f64>,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The length of the sample to generate (in tokens).
    #[arg(long, short = 'n', default_value_t = 100)]
    sample_len: usize,

    #[arg(long)]
    model_id: Option<String>,

    #[arg(long, default_value = "main")]
    revision: String,

    #[arg(long)]
    tokenizer_file: Option<String>,

    #[arg(long)]
    weight_files: Option<String>,

    // #[arg(long)]
    // quantized: bool,
    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,

    // Control vector related arguments.
    /// Control vector strength.
    #[arg(long, allow_negative_numbers = true, default_value_t = 0.)]
    control_strength: f32,

    /// Apply the control vector.
    #[arg(long, default_value_t = false)]
    apply_control_vector: bool,

    /// Path to read the control vector.
    #[arg(long, default_value = None)]
    control_vector_path: Option<String>,

    /// Path to read the control vector for suffixes.
    #[arg(long, default_value = None)]
    control_vector_path_sf: Option<String>,

    /// Base prompt to use for suffixes.
    #[arg(long)]
    base_prompt: Option<String>,

    /// Positive adjectives to use for suffixes.
    #[arg(long, value_delimiter = ',', num_args = 1..)]
    positive_adjectives: Vec<String>,

    /// Negative adjectives to use for suffixes.
    #[arg(long, value_delimiter = ',', num_args = 1..)]
    negative_adjectives: Vec<String>,

    /// Dump the last hidden states to a file.
    #[arg(long)]
    dump_last_hidden_states: bool,

    /// Apply PCA to control vector.
    #[arg(long)]
    pca_to_control_vector: bool,
}

fn householder(x: Tensor) -> Result<(Tensor, Tensor)> {
    let m = x.dims1()?;
    let alpha = x.get(0)?;
    let s = x.sqr()?.sum_all()?;
    let two = Tensor::from_slice(&[2.0f32], (), x.device())?;
    // if s is 0 return v = 0 and tau = 0
    let (v, tau) = if s.to_vec0::<f32>()? == 0.0 {
        let tau = Tensor::zeros((1,), x.dtype(), x.device())?;
        let v = Tensor::zeros((m,), x.dtype(), x.device())?;
        (v, tau)
    } else {
        let t = alpha.sqr()?.add(&s)?.sqrt()?;
        let alpha_le_zero_value = alpha.to_vec0::<f32>()?;

        let v0 = if alpha_le_zero_value == 1.0 {
            (alpha - t.clone())?
        } else {
            s.neg()?.clone().div(&(alpha + t.clone())?)?
        };
        let v0_sqr = v0.sqr()?;
        let tau = two.mul(&v0_sqr)?.div(&s.add(&v0_sqr)?)?;
        let v = x.slice_assign(&[0..1], &v0.unsqueeze(0)?)?;
        (v.div(&v0.repeat((m,))?)?, tau)
    };
    Ok((v, tau))
}

// aligned with the numpy implementation
fn householder_vectorized(x: Tensor) -> Result<(Tensor, Tensor)> {
    let linalg_norm_a = x.sqr()?.sum_all()?.sqrt()?;
    let a0 = x.get(0)?;
    let a0_plus_norm = a0.add(&linalg_norm_a)?;
    let v = x.div(&a0_plus_norm.repeat((x.dims1()?,))?)?;
    let v = v.slice_assign(&[0..1], &Tensor::from_slice(&[1.0f32], (1,), x.device())?)?;
    let v_sum = v.sqr()?.sum_all()?;
    let tau = Tensor::from_slice(&[2.0f32], (), x.device())?.div(&v_sum)?;
    Ok((v, tau))
}

// TODO: cleanup and and validate calculations
fn qr_decomposition(A: Tensor) -> Result<(Tensor, Tensor)> {
    let m = A.dims()[0];
    let n = A.dims()[1];
    let mut R = A.clone();
    let mut Q = Tensor::eye(m, DType::F32, &Device::Cpu)?;
    for j in 0..n {
        let r_slice = R.i((j..m, j..j + 1))?;
        let r_slice_flat = r_slice.flatten(0, 1)?;
        let (v, tau) = householder_vectorized(r_slice_flat)?;
        let h = Tensor::eye(m, DType::F32, &Device::Cpu)?;
        let empty_v = v.clone().reshape((1, m - j))?.repeat((m - j, 1))?;
        let v_t = empty_v.t()?;
        let v_mul_v_t = empty_v.mul(&v_t)?;
        let tau_v_mul_v_t = v_mul_v_t.clone().mul(&tau.repeat((m - j, m - j))?)?;
        let h = h.slice_assign(&[j..m, j..m], &h.i((j..m, j..m))?.sub(&tau_v_mul_v_t)?)?;
        R = h.matmul(&R)?;
        Q = h.matmul(&Q)?;
    }
    let S = R.i((0..n, 0..n))?;
    let mask = Tensor::triu2(n, DType::F32, &Device::Cpu)?;
    let R = S.mul(&mask)?;
    Ok((Q.i((0..n, 0..m))?.t()?, R))
}

fn eigen_qr_simple(x: Tensor) -> Result<(Tensor, Tensor)> {
    let iterations = 200;
    let mut Ak = x.clone();
    let n = x.dims()[0];
    let mut QQ = Tensor::eye(n, x.dtype(), &x.device())?;
    for _ in 0..iterations {
        let (Q, R) = qr_decomposition(Ak.clone())?;
        Ak = R.matmul(&Q)?;
        QQ = QQ.matmul(&Q)?;
    }
    // zero non-diagonal elements
    let indexes = Tensor::eye(n, x.dtype(), &x.device())?;
    // TODO: use a more efficient method reduce the matrix to its diagonal elements
    // sum across the first dimension to get the diagonal elements
    let E = Ak.mul(&indexes)?.sum(0)?;
    Ok((E, QQ))
}

// TODO: cleanup and and validate calculations
fn pca(x: Tensor) -> Result<(Tensor, Tensor, Tensor)> {
    // Step 1: Standardize the dataset
    let mean = x.mean(0)?;
    let mean_centered = x.sub(&mean.repeat((x.dims()[0], 1))?)?;
    let standardized = mean_centered;

    // Step 2: Compute the covariance matrix
    let cov = standardized.t()?.matmul(&standardized)?;
    let cov = cov.broadcast_div(&Tensor::from_slice(
        &[standardized.dims()[0] as f32 - 1.0f32],
        (1,),
        &Device::Cpu,
    )?)?;

    // Step 3: Compute the eigenvalues and eigenvectors
    let (eigenvalues, eigenvectors) = eigen_qr_simple(cov)?;

    // Step 4: Sort the eigenvectors by decreasing eigenvalues
    let mut sorted_eigenvalues = eigenvalues.to_vec1::<f32>()?;
    let mut argsort_indices: Vec<_> = (0..sorted_eigenvalues.len()).collect();
    argsort_indices.sort_by(|&i, &j| {
        sorted_eigenvalues[j]
            .partial_cmp(&sorted_eigenvalues[i])
            .unwrap()
    });

    // use the values to sort the vectors
    let mut sorted_eigenvectors = eigenvectors.to_vec2::<f32>()?;
    let mut sorted_eigenvectors = argsort_indices
        .iter()
        .map(|&i| sorted_eigenvectors[i].clone())
        .collect::<Vec<_>>();

    let sorted_eigenvectors = sorted_eigenvectors
        .iter()
        .flatten()
        .cloned()
        .collect::<Vec<f32>>();

    // convert back to tensor
    let sorted_eigenvectors = Tensor::from_vec(sorted_eigenvectors, (4, 4), &x.device())?;

    // Step 5: Transform the original matrix
    let transformed = standardized.matmul(&eigenvectors.t()?)?;

    Ok((transformed, eigenvalues, sorted_eigenvectors.t()?))
}

// TODO:
// - fix interface to follow intutive flow (capture, generate, apply)?
// - simplify modeling code and overall example
// - implement PCA to fufill `generate` function
// - move files into repo for easier access
fn main() -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();
    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };
    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle::utils::with_avx(),
        candle::utils::with_neon(),
        candle::utils::with_simd128(),
        candle::utils::with_f16c()
    );
    println!(
        "temp: {:.2} repeat-penalty: {:.2} repeat-last-n: {}",
        args.temperature.unwrap_or(0.),
        args.repeat_penalty,
        args.repeat_last_n
    );
    let mut complete_dataset = vec![];

    let mut sample_len = args.sample_len;

    if args.pca_to_control_vector {
        let (_transformed, _eigenvalues, eigenvectors) = pca(Tensor::from_slice(
            &[
                1., 2., 2., 3., //
                3., 4., 4., 5., //
                5., 6., 1., 0., //
                2., 1., 3., 2.0f32,
            ],
            (4, 4),
            &Device::Cpu,
        )?)?;

        let eigenvectors = eigenvectors.to_vec2::<f32>()?;

        // only print the first two vectors
        for vec in eigenvectors.iter().take(2) {
            println!("{:?}", vec);
        }
        // [0.5093906, 0.62907666, -0.30034626, -0.5045552]
        // [0.26052827, 0.5371797, 0.38106996, 0.7059383]

        // COMPARED TO PYTHON
        //
        // # Given tensor
        // tensor = np.array([
        //     [1.0, 2.0, 2.0, 3.0],
        //     [3.0, 4.0, 4.0, 5.0],
        //     [5.0, 6.0, 1.0, 0.0],
        //     [2.0, 1.0, 3.0, 2.0],
        // ])

        // pca = PCA(n_components=2, power_iteration_normalizer="QR", svd_solver="full")
        // pca.fit(tensor)
        // print(pca.components_)
        // [[ 0.50939062  0.6290765  -0.30034636 -0.50455527]
        //  [ 0.26052823  0.53717973  0.38106995  0.70593815]]

        return Ok(());
    }

    if args.dump_last_hidden_states {
        // read input prompts
        let suffixes = std::fs::read_to_string(
            "candle-examples/examples/mistral-control/all_truncated_outputs.json",
        )
        .unwrap();
        let suffixes: Vec<String> = serde_json::from_str(&suffixes).unwrap();
        let base = match args.base_prompt {
            Some(prompt) => prompt,
            None => "The weather is".to_string(),
        };

        // we only want to do one forward pass and capture the last hidden states
        sample_len = 1;

        let pos_adjs = args.positive_adjectives;
        let neg_adjs = args.negative_adjectives;

        let template = |adj: &str, suffix: &str| format!("{base} {adj} {suffix}");

        let mut pos_dataset = vec![];
        let mut neg_dataset = vec![];
        for suffix in &suffixes {
            for adj in &pos_adjs {
                pos_dataset.push(template(adj, suffix));
            }
            for adj in &neg_adjs {
                neg_dataset.push(template(adj, suffix));
            }
        }

        for i in 0..pos_dataset.len() {
            complete_dataset.push(pos_dataset[i].clone());
        }
        for i in 0..neg_dataset.len() {
            complete_dataset.push(neg_dataset[i].clone());
        }
    } else {
        complete_dataset.push(args.prompt.map_or("The weather is".to_string(), |p| p));
    }

    let control_vector: Vec<Vec<f32>> = match args.control_vector_path_sf {
        Some(path) => {
            // load safetensors file from path
            let data = candle::safetensors::load(path, &Device::Cpu)?;
            // get value at all_directions
            let all_directions = data.get("all_directions").unwrap();
            // convert to Vec<Vec<f32>>
            let all_directions = all_directions.to_vec2::<f32>()?;
            // scale the control vector by the control strength
            let all_directions = all_directions
                .iter()
                .map(|v| v.iter().map(|&v| v * args.control_strength).collect())
                .collect();
            all_directions
        }
        None => {
            println!("No control vector path provided");
            vec![]
        }
    };

    let start = std::time::Instant::now();
    let api = Api::new()?;
    let model_id = "mistralai/Mistral-7B-v0.1".to_string();
    let repo = api.repo(Repo::with_revision(
        model_id,
        RepoType::Model,
        args.revision,
    ));
    let tokenizer_filename = match args.tokenizer_file {
        Some(file) => std::path::PathBuf::from(file),
        None => repo.get("tokenizer.json")?,
    };
    let filenames = match args.weight_files {
        Some(files) => files
            .split(',')
            .map(std::path::PathBuf::from)
            .collect::<Vec<_>>(),
        None => candle_examples::hub_load_safetensors(&repo, "model.safetensors.index.json")?,
    };
    println!("retrieved the files in {:?}", start.elapsed());
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let start = std::time::Instant::now();
    let config = Config::config_7b_v0_1(args.use_flash_attn);
    let device = candle_examples::device(args.cpu)?;
    let (model, device) = {
        let dtype = if device.is_cuda() {
            DType::BF16
        } else {
            DType::F32
        };
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
        let model = Mistral::new(&config, vb)?;
        (Model::Mistral(model), device)
    };

    println!("loaded the model in {:?}", start.elapsed());

    let mut pipeline = TextGeneration::new(
        model,
        tokenizer,
        args.seed,
        args.temperature,
        args.top_p,
        args.repeat_penalty,
        args.repeat_last_n,
        control_vector,
        args.apply_control_vector,
        &device,
    );

    let mut last_hidden_states =
        Tensor::zeros((complete_dataset.len(), 32, 4096), DType::BF16, &device)?;

    pipeline.run(
        &mut last_hidden_states,
        complete_dataset.clone(),
        sample_len,
    )?;

    if args.dump_last_hidden_states {
        // save tensor to file
        let _ = last_hidden_states
            .save_safetensors("last_hidden_states", "new_last_hidden_states.safetensors");
    }

    Ok(())
}
