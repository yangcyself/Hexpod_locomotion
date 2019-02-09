# -^- coding: utf8 -^- 

# ENVIRONMENT = "TOY"
# ENVIRONMENT = "BLUE"
ENVIRONMENT = "VREP"

COMMENT="CRAZY"
FILEOUT = False
FILEOUT = FILEOUT and ENVIRONMENT=="TOY"

REFRESHTOPO = False
FUTHERTOPO = True
SETTEDTOPO = True
# MAP = "fence"
MAP = None
NOISE = False


#Configues for topomain
RESUME = 10500
LOGGING = False
OBSERVETOPO = True
TESTBEHAVE = True


#Configurations in toyrep
STRICT_BALANCE = True 
TIPS_order = False
TIPS_distance = True
DISPLAY = False

BLUEROBOT = ENVIRONMENT=="BLUE"

#Reward Options:
RWD_PAIN = True
RWD_DANEROUS = False


#Reward factors:
RWDFAC_PAIN = 0.01
RWDFAC_DANEROUS = 1
