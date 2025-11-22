# ----- STYLE TRANSFER PARAMETERS -----
num_steps = 300
style_weight = 1e6 -> very high when compared to content weight
content_weight = 1

style weight -> controls how much the style image influencesa the output. 
content weight -> controls how much the content image influences the output.

num_optimization -> more steps -> more refined stylizations
few steps -> incomplete stylization.

layers used for content and style losses:
content_layer_deafult = conv_4
style_layer_default = conv_1, conv_2, ...., conv_5

content layer -> deeper -> preserves more high -level structures
style layer -> fewer -> affects how texture is applied.

