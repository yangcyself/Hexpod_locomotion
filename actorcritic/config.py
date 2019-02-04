# -^- coding: utf8 -^- 

ENVIRONMENT = "TOY"
# ENVIRONMENT = "BLUE"
# ENVIRONMENT = "VREP"

COMMENT="CRAZY"
FILEOUT = False
FILEOUT = FILEOUT and ENVIRONMENT=="TOY"

REFRESHTOPO = False
FUTHERTOPO = True



#Configues for topomain
RESUME = 0
LOGGING = False
OBSERVETOPO = True
TESTBEHAVE = False

SETTEDTOPO = True

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
RWDFAC_PAIN = 1
RWDFAC_DANEROUS = 1
