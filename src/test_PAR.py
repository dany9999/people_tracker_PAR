
from . import PAR
from PIL import Image

# duke = PAR("duke")
# market = PAR("market")
# pred = {'backpack': None,
#          'bag': None, 
#          'handbag': None, 
#          'hat':None, 
#          'gender': None, 
#          'upblack': None, 
#          'upwhite': None, 
#          'upred': None, 
#          'uppurple': None, 
#          'upyellow': None, 
#          'upgray': None, 
#          'upblue':None, 
#          'upgreen': None,
#          'upbrown': None,
#          'uppink': None,
#          'uporange': None,
          
#          'downblack': None, 
#          'downwhite': None, 
#          'downpink': None, 
#          'downpurple': None, 
#          'downyellow': None, 
#          'downgray': None, 
#          'downblue': None, 
#          'downgreen': None, 
#          'downbrown': None,
#          '
#          'downred': None   
#         }



image = Image.open("train_image/11.jpg")

att_duke = duke.attribute_recognition(image)

att_market = market.attribute_recognition(image)

# print("Duke: {}".format(att_duke))

print("-----------------------------------------------------")

print("Market: {}".format(att_market))
print("-----------------------------------------------------")


# market_weights = {
#     "young": 0.356,
#     "teenager": 0.939,
#     "adult": 0.508,
#     "old": 0.019,  # F1 score for "old"
#     "backpack": 0.742,
#     "bag": 0.467,
#     "handbag": 0.104,
#     "clothes": 0.970,
#     "down": 0.959,
#     "up": 0.967,
#     "hair": 0.819,
#     "hat": 0.623,
#     "gender": 0.903,
#     "upblack": 0.823,
#     "upwhite": 0.863,
#     "upred": 0.871,
#     "uppurple": 0.755,
#     "upyellow": 0.865,
#     "upgray": 0.537,
#     "upblue": 0.566,
#     "upgreen": 0.750,
#     "downblack": 0.850,
#     "downwhite": 0.578,
#     "downpink": 0.788,
#     "downpurple": 0.9,  # non era definito perche accuracy = 1.0
#     "downyellow":0.1,   # da verificare in quanto era 0.0
#     "downgray": 0.559,
#     "downblue": 0.563,
#     "downgreen": 0.426,
#     "downbrown": 0.662
# }


# duke_weights = {
#     "backpack": 0.855,
#     "bag": 0.364,
#     "handbag": 0.126,
#     "boots": 0.787,
#     "gender": 0.817,
#     "hat": 0.768,
#     "shoes": 0.535,
#     "top": 0.463,
#     "upblack": 0.864,
#     "upwhite": 0.606,
#     "upred": 0.694,
#     "uppurple": 0.167,
#     "upgray": 0.432,
#     "upblue": 0.619,
#     "upgreen": 0.431,
#     "upbrown": 0.390,
#     "downblack": 0.772,
#     "downwhite": 0.522,
#     "downred": 0.689,
#     "downgray": 0.317,
#     "downblue": 0.703,
#     "downgreen": 0.7,  # da verificare in quanto accuracy = 0.97
#     "downbrown": 0.706
# }

# for label in duke_weights.keys():
#     att_duke[label] = att_duke[label] * duke_weights[label]
# print("duke weighted: {}".format(att_duke))

# # for label in market_weights.keys():
# #     att_market[label] = att_market[label] * market_weights[label]
# # print("market weighted: {}".format(att_market))



# # pred["duke"] 
