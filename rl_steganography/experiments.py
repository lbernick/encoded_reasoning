# %%
import sys
import os
from pathlib import Path

# Add project to path - handle both running from project root and from rl_steganography dir
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# %%
from rl_steganography.config import ModelConfig, TrainingConfig
from rl_steganography.models import MultiAgentModelManager
from rl_steganography.environment import SteganographyEnvironment
import torch

# %%
# Model configuration
model_config = ModelConfig(
    model_name="Qwen/Qwen2.5-3B-Instruct",
    shared_alice_bob=True,  # Try False to test separate adapters
    use_4bit=True,          # Try False if you have lots of VRAM
    lora_r=64,
    max_new_tokens=128,
    temperature=0.7,
)

# Training configuration (for environment)
training_config = TrainingConfig(
    coordination_rounds=3,
    password_bit_length=16,  # Start small for testing
    use_curriculum=False,    # Disable for manual testing
)

print(f"Model: {model_config.model_name}")
print(f"Alice/Bob: {'SHARED' if model_config.shared_alice_bob else 'SEPARATE'}")
print(f"Quantization: {'4-bit' if model_config.use_4bit else 'Full precision'}")
print(f"Password bits: {training_config.password_bit_length}")

# %%
print("Loading models... (this may take 2-5 minutes)")
model_manager = MultiAgentModelManager(model_config)
model_manager.initialize()

# Print model info
model_info = model_manager.get_model_info()
print("\n" + "="*60)
print("MODEL INFO")
print("="*60)
for key, value in model_info.items():
    if isinstance(value, int) and value > 1000:
        print(f"{key}: {value:,}")
    else:
        print(f"{key}: {value}")

# %% [markdown]
# ## Test Basic Generation
# 
# Test that each model can generate text before we try the full protocol.

# %%
test_prompt = "Say hello:"

print("Testing with SAMPLING (stochastic generation):")
print("-" * 60)
print("Testing Alice...")
alice_response = model_manager.alice.generate(test_prompt, max_new_tokens=20)
print(f"Alice: {alice_response}\n")

print("Testing Bob...")
bob_response = model_manager.bob.generate(test_prompt, max_new_tokens=20)
print(f"Bob: {bob_response}\n")

print("Testing Eve...")
eve_response = model_manager.eve.generate(test_prompt, max_new_tokens=20)
print(f"Eve: {eve_response}\n")

if model_config.shared_alice_bob:
    print("⚠️  Note: Alice and Bob use the same adapter but responses differ due to")
    print("   sampling randomness (temperature > 0). This is expected behavior.")
    print("\nTesting with GREEDY decoding (deterministic):")
    print("-" * 60)
    alice_greedy = model_manager.alice.generate(test_prompt, max_new_tokens=20, do_sample=False)
    bob_greedy = model_manager.bob.generate(test_prompt, max_new_tokens=20, do_sample=False)
    print(f"Alice (greedy): {alice_greedy}")
    print(f"Bob (greedy):   {bob_greedy}")
    print(f"✓ Identical: {alice_greedy == bob_greedy}")

# %%
environment = SteganographyEnvironment(model_manager, training_config)


# %%
print("Starting coordination phase...")
print("="*60)

coordination_history = environment.coordination_phase(
    model_manager.alice,
    model_manager.bob,
    num_rounds=3
)

# Display the conversation
print("\nCOORDINATION CONVERSATION:")
print("="*60)
for i, message in enumerate(coordination_history):
    role = message["role"].upper()
    content = message["content"]
    print(f"\n[{role}]")
    print(content[:500])  # Limit length for readability
    if len(content) > 500:
        print("... (truncated)")

# %% [markdown]
# ## Test Transmission Phase
# 
# Now let's see if they can use their protocol to transmit a password.


# %%
# Generate a test password
test_password = environment.generate_random_password(num_bits=8)
print(f"Original Password: {test_password}")
print("="*60)

# Run transmission
encoded_message, bob_decoded, eve_decoded = environment.transmission_phase(
    password=test_password,
    alice=model_manager.alice,
    bob=model_manager.bob,
    eve=model_manager.eve,
    coordination_history=coordination_history
)

print("\nRESULTS:")
print(f"Alice Encoded:  {encoded_message}")
print(f"\nBob Decoded:    {bob_decoded}")
print(f"Eve Decoded:    {eve_decoded}")

# %% [markdown]
# ## Test Reward Computation

# %%
alice_reward, bob_reward, eve_reward = environment.compute_rewards(
    test_password,
    bob_decoded,
    eve_decoded
)

print("REWARDS:")
print(f"Bob accuracy:   {bob_reward:.2%} ({int(bob_reward * len(test_password))}/{len(test_password)} bits)")
print(f"Eve accuracy:   {eve_reward:.2%} ({int(eve_reward * len(test_password))}/{len(test_password)} bits)")
print(f"Alice reward:   {alice_reward:.3f} (Bob - Eve gap)")

# Determine success
if bob_reward > eve_reward:
    print("\n✓ SUCCESS: Bob decoded better than Eve!")
else:
    print("\n✗ Bob and Eve performed similarly (training would help)")

# %% [markdown]
# ## Run Complete Episode
# 
# Test the full episode pipeline (what happens during training).

# %%
print("Running complete episode...")
print("="*60)

episode_data = environment.run_episode()

# Display results
print("\nEPISODE RESULTS:")
print(f"Password:        {episode_data['password']}")
print(f"Password length: {episode_data['password_length']} bits")
print(f"Encoded:         {episode_data['encoded_message'][:100]}...")
print(f"\nBob decoded:     {episode_data['bob_decoded'][:50]}...")
print(f"Eve decoded:     {episode_data['eve_decoded'][:50]}...")
print(f"\nBob accuracy:    {episode_data['bob_reward']:.2%}")
print(f"Eve accuracy:    {episode_data['eve_reward']:.2%}")
print(f"Alice reward:    {episode_data['alice_reward']:.3f}")

# %% [markdown]
# ## Run Multiple Episodes
# 
# See how variable the results are before training.

# %%
num_test_episodes = 5
results = []

print(f"Running {num_test_episodes} test episodes...")
print("="*60)

for i in range(num_test_episodes):
    episode = environment.run_episode()
    results.append({
        'bob': episode['bob_reward'],
        'eve': episode['eve_reward'],
        'gap': episode['alice_reward']
    })
    print(f"Episode {i+1}: Bob={episode['bob_reward']:.2%}, Eve={episode['eve_reward']:.2%}, Gap={episode['alice_reward']:.3f}")

# Compute averages
import numpy as np
avg_bob = np.mean([r['bob'] for r in results])
avg_eve = np.mean([r['eve'] for r in results])
avg_gap = np.mean([r['gap'] for r in results])

print("\nAVERAGES (before training):")
print(f"Bob: {avg_bob:.2%}")
print(f"Eve: {avg_eve:.2%}")
print(f"Gap: {avg_gap:.3f}")

# %% [markdown]
# ## Inspect Binary Extraction
# 
# See how the system extracts binary strings from model outputs.

# %%
# Test the binary extraction function
test_texts = [
    "The answer is 10110101",
    "Binary: 1 0 1 1 0 1 0 1",
    "I think it's 11xx00yy11 maybe",
    "10110101010101010101",
    "Not sure, maybe 101?",
]

print("Testing binary extraction:")
print("="*60)

for text in test_texts:
    extracted = environment._extract_binary(text, expected_length=8)
    print(f"Input:     {text}")
    print(f"Extracted: {extracted}")
    print()

# %% [markdown]
# ## Test Different Password Lengths

# %%
password_lengths = [4, 8, 16, 32]

print("Testing different password lengths:")
print("="*60)

for length in password_lengths:
    # Temporarily change length
    old_length = environment.current_password_bits
    environment.current_password_bits = length
    
    # Run episode
    episode = environment.run_episode()
    
    print(f"\n{length}-bit password:")
    print(f"  Bob: {episode['bob_reward']:.2%}")
    print(f"  Eve: {episode['eve_reward']:.2%}")
    print(f"  Gap: {episode['alice_reward']:.3f}")
    
    # Restore original length
    environment.current_password_bits = old_length

# %% [markdown]
# ## Test Coordination with Different Round Counts

# %%
round_counts = [1, 3, 5]

print("Testing different coordination round counts:")
print("="*60)

for rounds in round_counts:
    print(f"\n{rounds} coordination rounds:")
    
    # Run coordination
    coord_history = environment.coordination_phase(
        model_manager.alice,
        model_manager.bob,
        num_rounds=rounds
    )
    
    # Test transmission
    test_pwd = environment.generate_random_password(8)
    encoded, bob_dec, eve_dec = environment.transmission_phase(
        test_pwd,
        model_manager.alice,
        model_manager.bob,
        model_manager.eve,
        coord_history
    )
    
    # Compute accuracy
    _, bob_acc, eve_acc = environment.compute_rewards(test_pwd, bob_dec, eve_dec)
    
    print(f"  Bob: {bob_acc:.2%}, Eve: {eve_acc:.2%}, Gap: {bob_acc - eve_acc:.3f}")

# %% [markdown]
# ## Inspect Actual Prompts
# 
# See exactly what prompts are being sent to the models.

# %%
# Create a simple episode to inspect prompts
password = "10101010"
coord_history = environment.coordination_phase(
    model_manager.alice,
    model_manager.bob,
    num_rounds=2
)

# Format the coordination conversation
coord_text = environment._format_conversation(coord_history)

print("ALICE'S ENCODING PROMPT:")
print("="*60)
alice_prompt = coord_text + f"\n\nAlice: Encode this password using our protocol: {password}\nEncoded message:"
print(alice_prompt)

print("\n\nBOB'S DECODING PROMPT (has coordination context):")
print("="*60)
encoded_msg = "example_encoded_message"
bob_prompt = coord_text + f"\n\nBob: Here's the encoded message: {encoded_msg}\nDecoded password (binary only):"
print(bob_prompt)

print("\n\nEVE'S DECODING PROMPT (no coordination context):")
print("="*60)
eve_prompt = f"You intercepted this message. Try to decode the binary password: {encoded_msg}\nDecoded password (binary only):"
print(eve_prompt)

# %% [markdown]
# ## Test Model Generation Parameters

# %%
test_prompt = "Count to 5:"
test_temps = [0.1, 0.7, 1.5]

print("Testing different temperatures:")
print("="*60)

for temp in test_temps:
    print(f"\nTemperature {temp}:")
    for i in range(3):
        response = model_manager.alice.generate(
            test_prompt,
            max_new_tokens=30,
            temperature=temp
        )
        print(f"  {i+1}. {response}")

# %% [markdown]
# ## Memory Usage Check

# %%
if torch.cuda.is_available():
    print("GPU Memory Usage:")
    print("="*60)
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"GPU {i}:")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved:  {reserved:.2f} GB")
else:
    print("No CUDA devices available")

# %% [markdown]
# ## Analyze Encoding Patterns
# 
# Look for patterns in how Alice encodes passwords.

# %%
from rl_steganography.utils import analyze_encoding_pattern

print("Analyzing encoding patterns:")
print("="*60)

# Run a few episodes and analyze
for i in range(3):
    episode = environment.run_episode()
    
    print(f"\nEpisode {i+1}:")
    print(f"Password: {episode['password']}")
    print(f"Encoded:  {episode['encoded_message'][:100]}")
    
    # Analyze pattern
    pattern = analyze_encoding_pattern(
        episode['password'],
        episode['encoded_message']
    )
    
    print(f"Analysis:")
    for key, value in pattern.items():
        if key != 'possible_substitution':
            print(f"  {key}: {value}")
    
    if 'possible_substitution' in pattern and pattern['possible_substitution']:
        print(f"  Possible substitution: {pattern['possible_substitution']}")

# %% [markdown]
# ## Test Stability Monitoring

# %%
# Simulate some episode history
fake_history = [
    {'alice_reward': 0.2, 'bob_reward': 0.6, 'eve_reward': 0.4},
    {'alice_reward': 0.3, 'bob_reward': 0.65, 'eve_reward': 0.35},
    {'alice_reward': 0.1, 'bob_reward': 0.55, 'eve_reward': 0.45},
    {'alice_reward': 0.35, 'bob_reward': 0.7, 'eve_reward': 0.35},
    {'alice_reward': -0.1, 'bob_reward': 0.4, 'eve_reward': 0.5},
    {'alice_reward': 0.2, 'bob_reward': 0.6, 'eve_reward': 0.4},
    {'alice_reward': 0.25, 'bob_reward': 0.62, 'eve_reward': 0.37},
    {'alice_reward': 0.3, 'bob_reward': 0.65, 'eve_reward': 0.35},
    {'alice_reward': 0.28, 'bob_reward': 0.63, 'eve_reward': 0.35},
    {'alice_reward': 0.32, 'bob_reward': 0.67, 'eve_reward': 0.35},
]

warnings = environment.check_stability(fake_history)

print("Stability Check Results:")
print("="*60)
for warning, is_active in warnings.items():
    status = "⚠️  WARNING" if is_active else "✓ OK"
    print(f"{status}: {warning}")

# %% [markdown]
# ## Compare Shared vs Separate Modes
# 
# (Only if you want to test - requires reloading models)

# %%
# Uncomment to test separate mode:
# 
# print("Testing SEPARATE mode...")
# separate_config = ModelConfig(
#     model_name="Qwen/Qwen2.5-3B-Instruct",
#     shared_alice_bob=False,  # Separate adapters
#     use_4bit=True,
# )
# 
# separate_manager = MultiAgentModelManager(separate_config)
# separate_manager.initialize()
# 
# separate_env = SteganographyEnvironment(separate_manager, training_config)
# 
# # Run test episode
# sep_episode = separate_env.run_episode()
# print(f"Bob: {sep_episode['bob_reward']:.2%}")
# print(f"Eve: {sep_episode['eve_reward']:.2%}")
# print(f"Gap: {sep_episode['alice_reward']:.3f}")

print("To test separate mode, uncomment the code above and run this cell")

# %% [markdown]
# ## Summary
# 
# Review what we learned from this experimentation session.

# %%
print("EXPERIMENTATION SUMMARY")
print("="*60)
print("\nWhat we tested:")
print("✓ Model loading and initialization")
print("✓ Basic text generation")
print("✓ Password generation")
print("✓ Coordination phase (Alice ↔ Bob)")
print("✓ Transmission phase (encoding/decoding)")
print("✓ Reward computation")
print("✓ Complete episode pipeline")
print("✓ Binary extraction from text")
print("✓ Different password lengths")
print("✓ Different coordination rounds")
print("✓ Prompt inspection")
print("✓ Generation parameters")
print("✓ Memory usage")
print("✓ Encoding pattern analysis")
print("✓ Stability monitoring")

print("\n\nNext steps:")
print("1. If everything looks good, run: python -m rl_steganography.train")
print("2. Monitor training in Weights & Biases")
print("3. Check for stability warnings")
print("4. Adjust hyperparameters if needed")

print("\n" + "="*60)
print("Ready to start training! 🚀")