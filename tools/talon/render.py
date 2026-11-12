"""Offscreen renderer for skinned League meshes (moderngl). Raw file space is left-handed; we mirror X for display."""
import numpy as np, moderngl
from PIL import Image, ImageDraw
_ctx = None
def ctx():
    global _ctx
    if _ctx is None: _ctx = moderngl.create_standalone_context()
    return _ctx
VS = '''#version 330
uniform mat4 mvp; in vec3 in_pos; in vec3 in_nrm; in vec2 in_uv; in vec3 in_col; out vec3 n; out vec2 uv; out vec3 col;
void main(){ gl_Position = mvp*vec4(in_pos,1.0); n = in_nrm; uv = in_uv; col = in_col; }'''
FS = '''#version 330
uniform sampler2D tex; uniform int mode; uniform vec3 light; in vec3 n; in vec2 uv; in vec3 col; out vec4 f;
void main(){ vec3 nn = normalize(n); if(!gl_FrontFacing) nn = -nn; float d = 0.80 + 0.35*max(dot(nn, normalize(light)), 0.0);
 if(mode==0){ vec4 t = texture(tex, uv); if(t.a < 0.3) discard; f = vec4(pow(t.rgb*d, vec3(0.62)), 1.0);} else if(mode==1){ f = vec4(col*d,1.0);} else { f = vec4(col,1.0);} }'''
_prog = None
def prog():
    global _prog
    if _prog is None: _prog = ctx().program(vertex_shader=VS, fragment_shader=FS)
    return _prog
_texcache = {}
def gltex(img, key=None):
    if key is not None and key in _texcache: return _texcache[key]
    img = img.convert('RGBA'); t = ctx().texture(img.size, 4, img.tobytes()); t.build_mipmaps(); t.repeat_x = t.repeat_y = True
    if key is not None: _texcache[key] = t
    return t
def look(eye, tgt, up=(0, 1, 0)):
    eye, tgt, up = map(lambda v: np.array(v, float), (eye, tgt, up))
    f = tgt - eye; f /= np.linalg.norm(f); s = np.cross(f, up); s /= np.linalg.norm(s); u = np.cross(s, f)
    m = np.eye(4); m[0, :3] = s; m[1, :3] = u; m[2, :3] = -f; m[:3, 3] = -m[:3, :3] @ eye; return m
def ortho(hw, hh, n, f):
    m = np.eye(4); m[0, 0] = 1/hw; m[1, 1] = 1/hh; m[2, 2] = -2/(f-n); m[2, 3] = -(f+n)/(f-n); return m
def render(pos, nrm, uv, idx, parts, size=(512, 640), yaw=0.0, pitch=0.0, center=(0, 90, 0), half_h=110.0, bg=(96, 104, 120), lines=None, points=None):
    """parts: list of (index_start_tri, tri_count, PIL image or None, rgb color). yaw degrees about Y (0 = front view)."""
    c = ctx(); p = prog(); W, H = size
    fbo = c.framebuffer([c.renderbuffer((W, H), 4, samples=4)], c.depth_renderbuffer((W, H), samples=4)); fbo.use()
    c.enable(moderngl.DEPTH_TEST); c.disable(moderngl.CULL_FACE); c.clear(bg[0]/255, bg[1]/255, bg[2]/255, 1.0)
    P = np.array(pos, np.float32).copy(); N = np.array(nrm, np.float32).copy(); P[:, 0] *= -1; N[:, 0] *= -1
    cx = np.array(center, float).copy(); cx[0] *= -1
    ya, pa = np.radians(yaw), np.radians(pitch); d = 600.0
    eye = cx + d*np.array([np.sin(ya)*np.cos(pa), np.sin(pa), np.cos(ya)*np.cos(pa)])
    mvp = ortho(half_h*W/H, half_h, 1.0, 1500.0) @ look(eye, cx)
    p['mvp'].write(mvp.T.astype('f4').tobytes()); ld = eye - cx; ld = ld/np.linalg.norm(ld) + np.array([0.3, 0.5, 0.0]); p['light'].value = tuple(ld)
    for (ts, tc, img, color) in parts:
        tri = np.asarray(idx[ts:ts+tc], np.int32).reshape(-1)
        col = np.tile(np.array(color, np.float32)/255.0, (len(P), 1)) if np.ndim(color) == 1 else np.asarray(color, np.float32)
        vbo = c.buffer(np.hstack([P, N, np.array(uv, np.float32), col]).astype('f4').tobytes()); ibo = c.buffer(tri.tobytes())
        vao = c.vertex_array(p, [(vbo, '3f 3f 2f 3f', 'in_pos', 'in_nrm', 'in_uv', 'in_col')], ibo)
        if img is not None:
            gltex(img, id(img)).use(0); p['tex'].value = 0; p['mode'].value = 0
        else: p['mode'].value = 1
        vao.render(); vao.release(); vbo.release(); ibo.release()
    if lines is not None and len(lines):
        c.disable(moderngl.DEPTH_TEST)
        L = np.array(lines, np.float32).reshape(-1, 3).copy(); L[:, 0] *= -1
        col = np.tile(np.array([1.0, 0.9, 0.1], np.float32), (len(L), 1))
        vbo = c.buffer(np.hstack([L, np.zeros_like(L), np.zeros((len(L), 2), np.float32), col]).astype('f4').tobytes())
        vao = c.vertex_array(p, [(vbo, '3f 3f 2f 3f', 'in_pos', 'in_nrm', 'in_uv', 'in_col')]); p['mode'].value = 2; vao.render(moderngl.LINES); vao.release(); vbo.release()
    out = c.framebuffer([c.renderbuffer((W, H), 4)]); c.copy_framebuffer(out, fbo)
    img = Image.frombytes('RGBA', (W, H), out.read(components=4)).transpose(Image.FLIP_TOP_BOTTOM).convert('RGB')
    fbo.release(); out.release(); return img
def sheet(images, labels=None, cols=4, pad=4):
    w, h = images[0].size; rows = (len(images) + cols - 1)//cols
    out = Image.new('RGB', (cols*(w+pad)+pad, rows*(h+pad)+pad), (15, 15, 18)); d = ImageDraw.Draw(out)
    for i, im in enumerate(images):
        x = pad + (i % cols)*(w+pad); y = pad + (i//cols)*(h+pad); out.paste(im, (x, y))
        if labels: d.text((x+6, y+4), str(labels[i]), fill=(255, 255, 120))
    return out
def bone_lines(skel, world):
    ls = []
    for i, p in enumerate(skel.parents):
        if p >= 0: ls += [world[p, :3, 3], world[i, :3, 3]]
    return ls
