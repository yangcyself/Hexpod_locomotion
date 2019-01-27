#!/usr/bin python3
# import rospy
# from math import atan2,asin
# from tf2_msgs.msg import TFMessage
def callback(data):
	with open('vec_rot.txt', 'w') as f:
		f.write(str(data.transforms[-1].transform.translation.x)+" ")
		f.write(str(data.transforms[-1].transform.translation.y)+" ")
		f.write(str(data.transforms[-1].transform.translation.z)+" ")
		x=data.transforms[-1].transform.rotation.x
		y=data.transforms[-1].transform.rotation.y
		z=data.transforms[-1].transform.rotation.z
		w=data.transforms[-1].transform.rotation.w
		f.write(str(atan2(2*(y*z+w*x), w*w-x*x-y*y+z*z))+" ")
		f.write(str(asin(-2*(x*z-w*y)))+" ")
		f.write(str(atan2(2*(x*y+w*z), w*w+x*x-y*y-z*z))+" ")
def listener():
	# rospy.init_node('listener', anonymous=True) 
	# rospy.Subscriber("/tf",TFMessage, callback) 
	# rospy.spin()
	pass
 
listener()
if __name__ == '__main__':
	listener()
