## Adaptive Persona Context Modulation for Emotional Support Conversation
![APCM](https://github.com/hurricanewhk/APCM/blob/master/APCM.jpg)
### Abstract

Personalized Emotion Support Conversation (ESC) systems assist users (i.e.seekers) in navigating negative emotional states through personalized, empathetic interactions. While existing systems attempt to infer persona from dialogue to understand seekers, they often struggle to distinguish supporter's and seeker's roles and utilize seeker's persona appropriately. We present a novel **A**daptive **P**ersona **C**ontext **M**odulation approach, APCM, that dynamically adjusts a weight balance between persona and context in different scenarios. APCM addresses two key challenges in emotional support conversations. First, it more comprehensively captures the seeker's persona from the dialogue history while avoiding role confusion. Second, it introduces an adaptive modulation module that dynamically balances the influence of persona and context information during response generation, better reflecting real-world conversation patterns where persona relevance varies across utterances.
Extensive experiments on benchmark datasets demonstrate the effectiveness of APCM, achieving state-of-the-art performance in emotional support dialogue generation.

### Codes
Our code (in codes) mainly references [https://github.com/chengjl19/PAL/tree/main](https://github.com/chengjl19/PAL/tree/main)

### Data process
```
python codes/_reformat/process.py --add_persona True
```
### Model training

You should first download the [BlenderBot-small](https://huggingface.co/facebook/blenderbot_small-90M) model and put the pytorch_model.bin file in Blenderbot_small-90M.

Then run ```RUN/prepare_strat.sh```

And run ```RUN/train_strat.sh```

### Model inference
run ```RUN/infer_strat.sh```
