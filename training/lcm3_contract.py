"""Fixed LCM3 readout-qualification settings; no compute on import."""
ARMS=('direct_normalized','bridge_normalized','bridge_raw','direct_no_write')
CONTROLS=('full','reset','shuffle','trained_no_write','bridge_normalized','bridge_raw','initial')
PARENT_MODEL_HASHES=(
    '525fd4a8a928f831c9628f09d6ad8e7ecaee8307da49bb0e588064c297cf52ef',
    'db393501eeb9b8ea1ac92c02f88531fb03ffee43b2ee988981566a3b26b359bb',
    '264517dea0838616b332d6520e3c4e6fcb752d1c463275d92e1dff907c0f1a7d',
    'd3d17a208c2614000d0e893f8eeb1741fa3fd3a5c40c79f8d48321fb2b43046b')
CONFIG=dict(trials=4,batch=64,updates=384,train_delays=(8,16),evaluation_delays=(64,128),
    normalization_seed=205611000,normalization_n=1024,normalization_floor=1e-5,
    initialization_base=206212000,training_base=205612000,evaluation_base=205712000,
    evaluation_action_base=206412000,lr=.03,clip=1.,evaluation_n=512,
    bootstrap_seed=206512000,bootstrap_draws=10000,accuracy_min=.90,recall_min=.90,
    reset_effect_min=.30,shuffle_effect_min=.60,opposite_direction_min=.80,
    interface_effect_min=.10,scaling_effect_min=.10,arms=ARMS,controls=CONTROLS,twins=('a','b'))
