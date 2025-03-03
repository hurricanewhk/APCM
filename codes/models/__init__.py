
from models.strat_blenderbot_small import Model as strat_blenderbot_small
from models.vanilla_blenderbot_small import Model as vanilla_blenderbot_small
from models.strat_blenderbot_small_no_persona import Model as strat_blenderbot_small_no_persona
from models.strat_dialogpt import Model as strat_dialogpt
from models.vanilla_dialogpt import Model as vanilla_dialogpt
from models.gated_encoder_blenderbot_small import Model as gated_encoder_blenderbot_small
from models.gated_dpr_blenderbot_small import Model as gated_dpr_blenderbot_small

models = {
    
    'vanilla_blenderbot_small': vanilla_blenderbot_small,
    'strat_blenderbot_small': strat_blenderbot_small,
    'strat_blenderbot_small_no_persona':strat_blenderbot_small_no_persona,
    'vanilla_dialogpt': vanilla_dialogpt,
    'strat_dialogpt': strat_dialogpt,
    'gated_encoder_blenderbot_small': gated_encoder_blenderbot_small,
    'gated_dpr_blenderbot_small': gated_dpr_blenderbot_small,
}