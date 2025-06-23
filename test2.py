from realhf.system.push_pull_stream import ZMQJsonPuller, ZMQJsonPusher

pusher = ZMQJsonPuller()
print("success")
pusher.close()
