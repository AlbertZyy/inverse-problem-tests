
from fealpy.backend import backend_manager as bm
from viztracer import VizTracer

bm.set_backend('numpy')

if bm.backend_name == 'pytorch':
    bm.set_default_device('cuda:0')

tracer = VizTracer(output_file='test_backend_matmul.json')
tracer.start()

SIZE = 100
A = bm.random.randn(SIZE, SIZE)
F = bm.random.randn(SIZE, SIZE)

u = A @ F

tracer.stop()
tracer.save()
