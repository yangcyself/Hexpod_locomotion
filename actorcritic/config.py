# -^- coding: utf8 -^- 

ENVIRONMENT = "TOY"
# ENVIRONMENT = "BLUE"
# ENVIRONMENT = "VREP"

COMMENT="TEST"
FILEOUT = False

#Configues for topomain
RESUME = 2900
LOGGING = False
OBSERVETOPO = True

#Configurations in toyrep
STRICT_BALANCE = True 
TIPS_order = True
TIPS_distance = True
DISPLAY = False

BLUEROBOT = ENVIRONMENT=="BLUE"

#Reward Options:
RWD_PAIN = True



#Reward factors:
RWDFAC_PAIN = 1
