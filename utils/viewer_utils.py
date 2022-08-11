from contextlib import contextmanager
import numpy as np
import torch
import torch.nn.functional as F
import pycuda.driver
from pycuda.gl import graphics_map_flags
from glumpy import app, gloo, gl


# https://gist.github.com/victor-shepardson/3eb67c3664cde081a7e573376b1b0b54

@contextmanager
def cuda_activate(img):
    """Context manager simplifying use of pycuda.gl.RegisteredImage"""
    mapping = img.map()
    yield mapping.array(0, 0)
    mapping.unmap()


def create_shared_texture(w, h, c=4,
                          map_flags=graphics_map_flags.WRITE_DISCARD,
                          dtype=np.uint8):
    """Create and return a Texture2D with gloo and pycuda views."""
    tex = np.zeros((h, w, c), dtype).view(gloo.Texture2D)
    tex.activate()  # force gloo to create on GPU
    tex.deactivate()
    cuda_buffer = pycuda.gl.RegisteredImage(
        int(tex.handle), tex.target, map_flags)
    return tex, cuda_buffer


class Viewer:
    def __init__(self, width: int = 512, height: int = 512):
        # create window with OpenGL context
        app.use('glfw')
        self.window = app.Window(width, height, fullscreen=False)
        self.window.on_draw = self.on_draw
        self.window.on_close = self.on_close
        self.window.on_mouse_drag = self.on_mouse_drag
        self.window.on_mouse_press = self.on_mouse_press
        self.window.on_mouse_release = self.on_mouse_release
        self.window.on_mouse_scroll = self.on_mouse_scroll
        self.draw_callback = None
        self.mouse_press_callback = None
        self.mouse_release_callback = None
        self.mouse_drag_callback = None
        self.mouse_scroll_callback = None

    def run(self):
        self.setup()
        app.run()

    def setup(self):
        w, h = self.window.get_size()
        import pycuda.gl.autoinit
        import pycuda.gl
        # setup pycuda and torch
        assert torch.cuda.is_available()
        print('using GPU {}'.format(torch.cuda.current_device()))
        # torch.nn layers expect batch_size, channels, height, width
        self.state = torch.cuda.FloatTensor(1, 3, h, w)
        self.state.uniform_()
        # create a buffer with pycuda and gloo views
        tex, self.cuda_buffer = create_shared_texture(w, h, 4)
        # create a shader to program to draw to the screen
        vertex = """
        uniform float scale;
        attribute vec2 position;
        attribute vec2 texcoord;
        varying vec2 v_texcoord;
        void main()
        {
            v_texcoord = texcoord;
            gl_Position = vec4(scale*position, 0.0, 1.0);
        } """
        fragment = """
        uniform sampler2D tex;
        varying vec2 v_texcoord;
        void main()
        {
            gl_FragColor = texture2D(tex, v_texcoord);
        } """
        # Build the program and corresponding buffers (with 4 vertices)
        self.screen = gloo.Program(vertex, fragment, count=4)
        # Upload data into GPU
        self.screen['position'] = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
        self.screen['texcoord'] = [(0, 0), (0, 1), (1, 0), (1, 1)]
        self.screen['scale'] = 1.0
        self.screen['tex'] = tex

    def on_draw(self, dt):
        self.window.set_title(str(self.window.fps).encode("ascii"))
        tex = self.screen['tex']
        h, w = tex.shape[:2]
        if self.draw_callback is not None:
            tensor = self.draw_callback(dt, h, w)
        else:
            tensor = torch.zeros((h, w, 4), dtype=torch.float32, device='cuda') + 0.5
        tensor = (255 * tensor).byte().contiguous()  # convert to ByteTensor
        assert tex.nbytes == tensor.numel() * tensor.element_size()
        with cuda_activate(self.cuda_buffer) as ary:
            cpy = pycuda.driver.Memcpy2D()
            cpy.set_src_device(tensor.data_ptr())
            cpy.set_dst_array(ary)
            cpy.width_in_bytes = cpy.src_pitch = cpy.dst_pitch = tex.nbytes // h
            cpy.height = h
            cpy(aligned=False)
            torch.cuda.synchronize()
        # draw to screen
        self.window.clear()
        self.screen.draw(gl.GL_TRIANGLE_STRIP)

    def on_mouse_press(self, x, y, button):
        if self.mouse_press_callback:
            self.mouse_press_callback(x, y, button)

    def on_mouse_release(self, x, y, button):
        if self.mouse_release_callback:
            self.mouse_release_callback(x, y, button)

    def on_mouse_drag(self, x, y, dx, dy, buttons):
        if self.mouse_drag_callback:
            self.mouse_drag_callback(x, y, dx, dy, buttons)

    def on_mouse_scroll(self, x, y, dx, dy):
        if self.mouse_scroll_callback:
            self.mouse_scroll_callback(x, y, dx, dy)

    def on_close(self):
        pass
        # pycuda.gl.autoinit.context.pop()
