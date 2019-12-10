# RL-Music-Tuner
Music generation and tuning using convolutional networks and reinforcement learning

Required Packages:
- PyTorch
- Mido

To Use:
--------
Our demo allows you to generate MIDI samples with a given policy derived from a pre-trined Q-network. The model is primed with a random starting state then generates a drum composition with 32 sixteenth notes for a total of two measures.

Run with: demo.exe [model] [num_samples]


### Parameter Options:

#### model: 
- "Note_CNN" - supervised model
- "0.01" - Q-network trained with epsilon = 0.01 greedy action selection
- "0.05" - epsilon = 0.05
- "0.1" - epsilon = 0.1
- "0.3" - epsilon = 0.3
- "0.5" - epsilon = 0.5
  
#### num_samples: 
- integer, the number of MIDI samples you wish to generate

### Outputs:
- num_samples MIDI samples with filenames "demo_song-#.mid"
