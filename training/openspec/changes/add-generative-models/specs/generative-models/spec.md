# Generative Models

## ADDED Requirements

### Requirement: VAE Architecture for Segment Generation
The system SHALL provide a variational autoencoder that generates segments conditioned on chromatic mode.

#### Scenario: Encode to latent distribution
- **WHEN** a segment is encoded
- **THEN** mean and log-variance parameters of a Gaussian distribution are produced

#### Scenario: Reparameterization sampling
- **WHEN** sampling from latent distribution during training
- **THEN** reparameterization trick enables backpropagation through sampling

#### Scenario: Conditional decoding
- **WHEN** decoding a latent vector with target chromatic mode
- **THEN** a segment exhibiting that chromatic mode is generated

#### Scenario: KL divergence regularization
- **WHEN** training VAE
- **THEN** KL divergence between learned and prior distributions is minimized

### Requirement: Diffusion Model for High-Quality Generation
The system SHALL provide a diffusion model that generates segments through iterative denoising.

#### Scenario: Forward diffusion adds noise
- **WHEN** applying forward diffusion
- **THEN** Gaussian noise is progressively added to segments over T steps

#### Scenario: Reverse diffusion denoises
- **WHEN** generating from noise
- **THEN** learned denoising network iteratively removes noise to produce a segment

#### Scenario: Conditional generation
- **WHEN** generating with chromatic mode conditioning
- **THEN** conditioning signal guides denoising toward target mode

#### Scenario: Sampling schedule
- **WHEN** using DDPM or DDIM sampling
- **THEN** configurable step count trades off quality and speed

### Requirement: Autoregressive Transformer Generation
The system SHALL provide a GPT-style model that generates segments token-by-token.

#### Scenario: Causal autoregressive generation
- **WHEN** generating sequences
- **THEN** each token is predicted conditional on all previous tokens

#### Scenario: Prompt-based conditioning
- **WHEN** given a prompt like "Generate a RED segment with high temporal complexity"
- **THEN** generation follows the conditioning prompt

#### Scenario: Sampling strategies
- **WHEN** sampling tokens
- **THEN** temperature, top-k, and nucleus (top-p) sampling control diversity

#### Scenario: Multi-modal tokenization
- **WHEN** generating text, audio, and MIDI
- **THEN** unified token vocabulary spans all modalities

### Requirement: Latent Space Manipulation
The system SHALL enable controlled manipulation of latent representations to adjust generation attributes.

#### Scenario: Sample from prior
- **WHEN** sampling VAE latent vectors
- **THEN** vectors are drawn from the learned prior distribution

#### Scenario: Attribute-specific manipulation
- **WHEN** adjusting intensity in latent space
- **THEN** learned directions increase or decrease rebracketing intensity

#### Scenario: Latent interpolation
- **WHEN** interpolating between two segment latents
- **THEN** smooth transitions in generation are produced

### Requirement: Audio Tokenization
The system SHALL tokenize audio waveforms into discrete tokens for generative modeling.

#### Scenario: Neural codec encoding
- **WHEN** using EnCodec or similar
- **THEN** audio is compressed to discrete tokens with reconstruction

#### Scenario: Detokenization
- **WHEN** generating audio tokens
- **THEN** neural decoder reconstructs waveforms from tokens

### Requirement: MIDI Tokenization
The system SHALL tokenize MIDI events into discrete tokens for generative modeling.

#### Scenario: REMI tokenization
- **WHEN** using REMI (Revamped MIDI-derived events)
- **THEN** MIDI is represented as bar, position, pitch, velocity, duration tokens

#### Scenario: Compound Word tokenization
- **WHEN** using Compound Word approach
- **THEN** multiple event attributes are combined into single tokens

### Requirement: Generation Quality Evaluation
The system SHALL assess the quality of generated segments using quantitative metrics.

#### Scenario: Fréchet Inception Distance
- **WHEN** computing FID
- **THEN** distance between real and generated segment distributions is measured

#### Scenario: Diversity metrics
- **WHEN** evaluating diversity
- **THEN** intra-class variance quantifies generation variety

#### Scenario: Reconstruction quality
- **WHEN** evaluating VAE
- **THEN** reconstruction error (MSE, perceptual loss) is measured

### Requirement: Chromatic Consistency Evaluation
The system SHALL verify that generated segments match target chromatic modes.

#### Scenario: Style classifier verification
- **WHEN** generating a RED segment
- **THEN** a style classifier confirms it is recognized as RED

#### Scenario: Consistency rate
- **WHEN** generating many segments
- **THEN** percentage matching target chromatic mode is reported

### Requirement: Guided Sampling
The system SHALL support classifier-guided generation to steer outputs toward desired attributes.

#### Scenario: Classifier guidance
- **WHEN** using a trained rebracketing classifier
- **THEN** generation is guided toward high rebracketing probability

#### Scenario: Gradient-based steering
- **WHEN** applying classifier gradients during sampling
- **THEN** intermediate outputs are adjusted toward target attributes

### Requirement: Generation Configuration
The system SHALL provide comprehensive configuration for generative models.

#### Scenario: Model type selection
- **WHEN** config.model.generative.type is "vae", "diffusion", or "autoregressive"
- **THEN** the corresponding model is instantiated

#### Scenario: Sampling parameters
- **WHEN** config.generation.sampling specifies temperature=0.8, top_k=50
- **THEN** sampling uses those parameters

#### Scenario: Latent dimension
- **WHEN** config.model.vae.latent_dim is set to 512
- **THEN** VAE latent space has 512 dimensions
