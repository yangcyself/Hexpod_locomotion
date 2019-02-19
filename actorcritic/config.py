# -^- coding: utf8 -^- 

# ENVIRONMENT = "TOY"
ENVIRONMENT = "BLUE"
# ENVIRONMENT = "VREP"

COMMENT="BLUE"
FILEOUT = False
FILEOUT = FILEOUT and ENVIRONMENT=="TOY"

REFRESHTOPO = True
FUTHERTOPO = False
# SETTEDTOPO = True
SETTEDTOPO = False
DISPLAY_OBS=True
# MAP = "fence"
MAP = None
NOISE = False
SETMAP = True
CLIP = True
assert((SETMAP and not SETTEDTOPO )or (not SETMAP and SETTEDTOPO))


#Configues for topomain
RESUME = 9200
# RESUME = "savior/2700"
LOGGING = False
OBSERVETOPO = True
TESTBEHAVE = True


#Configurations in toyrep
STRICT_BALANCE = True 
TIPS_order = False
TIPS_distance = True
DISPLAY = False

BLUEROBOT = ENVIRONMENT=="BLUE"
import platform
    
FCNTL = platform.system()=="Linux"


#Reward Options:
RWD_PAIN = True
RWD_DANEROUS = False


#Reward factors:
RWDFAC_PAIN = 0.01
RWDFAC_DANEROUS = 1
