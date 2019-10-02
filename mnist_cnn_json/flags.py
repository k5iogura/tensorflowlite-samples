global __floating_infer
global __relux_info
class FLAGs:
    def __init__(initval=True):
        global __floating_infer
        global __relux_info
        __floating_infer = True
        __relux_info     = False

#
# Floating inference mode cntorl
    @property
    def floating_infer(self):
        global __floating_infer
        return(__floating_infer)

    @floating_infer.setter
    def floating_infer(self,val):
        global __floating_infer
        __floating_infer = val

#
# Debug info control for RELUx
    @property
    def relux_info(self):
        global __relux_info
        return __relux_info

    @relux_info.setter
    def relux_info(self,val):
        global __relux_info
        __relux_info = val

flags = FLAGs()
