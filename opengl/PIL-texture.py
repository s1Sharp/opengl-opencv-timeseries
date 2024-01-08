from PIL import Image
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *


def load_texture(file):
    image = Image.open(file)
    image_bytes = image.tobytes('raw', 'RGB', 0, -1)
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.size[0], image.size[1], 0, GL_RGB, GL_UNSIGNED_BYTE, image_bytes)
    return texture_id


def draw_object():
    glBegin(GL_TRIANGLES)
    glTexCoord2f(0.0, 0.0)
    glVertex3f(-0.5, -0.5, 0.0)
    glTexCoord2f(1.0, 0.0)
    glVertex3f(0.5, -0.5, 0.0)
    glTexCoord2f(1.0, 1.0)
    glVertex3f(0.5, 0.5, 0.0)
    glEnd()


def display():
    glClear(GL_COLOR_BUFFER_BIT)
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    draw_object()
    glDisable(GL_TEXTURE_2D)
    glutSwapBuffers()


glutInit()
glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE)
glutInitWindowSize(640, 480)
glutCreateWindow('Timur')
glutDisplayFunc(display)

texture_id = load_texture('texture.jpg')

glClearColor(0.0, 0.0, 0.0, 0.0)
glMatrixMode(GL_PROJECTION)
glLoadIdentity()
gluOrtho2D(-1.0, 1.0, -1.0, 1.0)
glMatrixMode(GL_MODELVIEW)
glLoadIdentity()

glutMainLoop()