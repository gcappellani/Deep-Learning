class ShapeException(Exception):
    def __init__(self, in_shape, kernel_shape, stride):
        super().__init__("Resulting output shape is not valid for in_shape=" + str(in_shape) + \
                         " kernel_shape=" + str(kernel_shape) + " stride=" + str(stride))
