# SEQ2SEQ Piano Transcription
- Tokenization is based on the [Magenta Team](https://magenta.tensorflow.org/)'s work 
- Can we perform piano transcription using Listen Attend and Spell model?

## Data
- [MAESTRO dataset V1.0](https://magenta.tensorflow.org/datasets/maestro)
- 512 Mel bin with 10ms Resolution 
- MIDI tokenziation with absolute time shift for each segments
- Vocab contains velocity, Note and time stamp

## Model Structure
### Encoder
- 4 Layers of BI-LSTM 
### Decoder
- Additive attention with 1 Layers of Uni-LSTM with linear-layer 

## Result
### Training curve
![Training Curve](https://user-images.githubusercontent.com/47840814/172285360-54de230a-6f5b-4edc-ac14-ece7bbbabfa4.png)

### F-1 Socre result
![Result](https://user-images.githubusercontent.com/47840814/172285068-a79f4e5f-ec3a-41d7-b54b-ae7513e76ae3.png)

## TO-DO
- Requries code Refactoring
- This repo is not final version
