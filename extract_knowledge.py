import logging
import sys
import os
import random
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer, 
    TrainingArguments, 
    AutoModelWithLMHead,
    AutoTokenizer,
    pipeline
)

import pathlib
import textwrap

import google.generativeai as genai

GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro')
# from keybert import KeyBERT

# keyword_generator = pipeline('text-generation',model='gpt2', tokenizer='gpt2')

prompt = "Generate 5 keywords to summarize Minecraft wiki: The badlands, formerly and commonly referred to as the mesa biome, are uncommon warm biomes, featuring mounds of terracotta, coming in various colors. There are three different variants of the badlands biome.\nBadlands biomes feature large mounds of terracotta, in layers of various colors; specifically, red, orange, yellow, white, light gray, brown, and regular terracotta may all naturally generate. These mounds rise up from a layer of red sand, and are about 10-15 blocks thick, giving way to typical stone variants below that. Cactus and dead bushes generate frequently across the landscape, similarly to deserts. Badlands biomes generate in warm regions and in areas with low erosion, which means that they usually generate in areas with plateaus and sometimes even mountain peaks.\nMineshafts generate at much higher elevations in these biomes, often exposed to fresh air. Their supporting planks and fences are made of dark oak rather than oak. Additionally, gold ore generates up to elevation Y=255 (rather than the normal Y=32), and at much higher rates than the rest of the world, making badlands tunnels excellent sources of gold.\nWhile all badlands biomes are rich in unique building materials and gold ore, there are no passive mobs. Furthermore, trees, grass, and water are uncommon, so food cultivation can be difficult. Trees and grass appear only atop wooded badlands.\nBeing a dry biome, it never rains, meaning lightning strikes are impossible. The exceptions are the rivers that cut through the badlands, where it can still rain and cast lightning. The usual darkening of the sky and hostile mob spawning that accompany thunderstorms still occurs during inclement weather.\nThe colors of specific terracotta strata (bands, layers) in these biomes are the same throughout all badlands biomes for any particular world. This means, for example, a layer of white terracotta might generate between the lines (X=200, Y=71) and (X=400, Y=72), being the same for all changes in Z. One may note chunk patterns when strata jump one Y-level at a particular X value. Often, the topmost layer of stained terracotta is replaced by regular terracotta, most often on plateau tops. Additionally, the topmost layer of terracotta below Y level 63 is always orange in color.\nIn Bedrock Edition, seeds have 32 bits, so strata are completely different compared to Java Edition, even if using the same equivalent seed number. Terracotta strata are still the same throughout all badlands biomes in a given world/seed. The elevation of a particular stratum is randomly varied but no more than a few y-layers.\nFor variants exist before 1.18, see Biome/Before 1.18.\nThere are three badlands biome variants (badlands, wooded badlands and eroded badlands), with three removed variants (badlands plateau, modified badlands plateau, and modified wooded badlands plateau) in previous versions.\nThe ordinary badlands biome has red sand as a topsoil layer and large mounds of terracotta in various colors. The sand and terracotta give way to stone and ores a few layers down. Cacti and dead bushes dot the sands. Additionally, can generate in taller and more jagged and pointy peaks that often pass the clouds and can peak at y=256.\nThe wooded badlands generates groves of trees at high altitudes. The uppermost layers of terrain have large patches of grass and coarse dirt, with oak trees growing on them. Here, the grass and oak leaves take on a dull greenish-brown color darker than that of the Savanna biome, giving it a droughted appearance; additionally, all naturally generated trees are small variants. This variant is a rare source of wood in the otherwise barren badlands. The forest begins generating above elevations of roughly Y=100. This variant generates at higher humidity values compared to the default badlands, which means that it can often be found bordering jungles or having lush caves underground.\nWooded badlands use the same mob spawning chances as badlands.\nThe eroded badlands features unique formations of terracotta hoodoos, narrow spires that rise up from the red sand floor of the biome's drainage basins. In Bedrock Edition, passive mobs can spawn here. This biome is intended to resemble the famous Bryce Canyon in Utah, USA, which features hoodoos across its landscape.\nIn Java Edition, eroded badlands use the same mob spawning chances as badlands.\nIn Bedrock Edition, eroded badlands use the same mob spawning chances as badlands for hostile categories. As for the others:\nRegular oak mineshafts do not generate in the badlands biomes, but they can generate into one if a neighboring biome generates the mineshaft.\nAlthough dark oak mineshafts generate in badlands, dark oaks do not grow in this biome.\nEroded badlands continue to generate terrain in the ""nothingness"" part of the Far Lands. Other badlands biomes are an ocean down to bedrock layer, like most biomes."
# prompt = prompt[:1000]
# result = keyword_generator(prompt, max_new_tokens=256)[0]['generated_text']
response = model.generate_content(prompt).text

print(response)