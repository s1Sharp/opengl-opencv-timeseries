import pygame
import random
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *

'''
На примере готового кубик-рубика рассмотреть обработку клавиш для вращения граней,
прорисовку quad как массива кубиков, заливку граней разными цветами.

*Добавить клавиши для вращения кубик-рубика размером 4*4.
'''

vertices = (
    ( 1, -1, -1), ( 1,  1, -1), (-1,  1, -1), (-1, -1, -1),
    ( 1, -1,  1), ( 1,  1,  1), (-1, -1,  1), (-1,  1,  1)
)
edges = ((0,1),(0,3),(0,4),(2,1),(2,3),(2,7),(6,3),(6,4),(6,7),(5,1),(5,4),(5,7))
surfaces = ((0, 1, 2, 3), (3, 2, 7, 6), (6, 7, 5, 4), (4, 5, 1, 0), (1, 5, 7, 2), (4, 0, 3, 6))
colors = ((.8, .1, .3), (.1, .8, .3), (.8, .4, .1), (.8, .8, .1), (.8, .8, .8), (.1, .3, .8))

class Cube():
    def __init__(self, id, N, scale):
        self.N = N
        self.scale = scale
        self.init_i = [*id]
        self.current_i = [*id]
        self.rot = [[1 if i==j else 0 for i in range(3)] for j in range(3)]

    def isAffected(self, axis, slice, dir):
        return self.current_i[axis] == slice

    def update(self, axis, slice, dir):

        if not self.isAffected(axis, slice, dir):
            return

        i, j = (axis+1) % 3, (axis+2) % 3
        for k in range(3):
            self.rot[k][i], self.rot[k][j] = -self.rot[k][j]*dir, self.rot[k][i]*dir

        self.current_i[i], self.current_i[j] = (
            self.current_i[j] if dir < 0 else self.N - 1 - self.current_i[j],
            self.current_i[i] if dir > 0 else self.N - 1 - self.current_i[i] )

    def transformMat(self):
        scaleA = [[s*self.scale for s in a] for a in self.rot]  
        scaleT = [(p-(self.N-1)/2)*2.1*self.scale for p in self.current_i] 
        return [*scaleA[0], 0, *scaleA[1], 0, *scaleA[2], 0, *scaleT, 1]

    def draw(self, col, surf, vert, animate, angle, axis, slice, dir):

        glPushMatrix()
        if animate and self.isAffected(axis, slice, dir):
            glRotatef( angle*dir, *[1 if i==axis else 0 for i in range(3)] )
        glMultMatrixf( self.transformMat() )

        glBegin(GL_QUADS)
        for i in range(len(surf)):
            glColor3fv(colors[i])
            for j in surf[i]:
                glVertex3fv(vertices[j])
        glEnd()

        glLineWidth(3.0)
        for i in range(len(surf)):
            glBegin(GL_LINE_LOOP)
            glColor3fv((0.0,0.0,0.0))
            for j in surf[i]:
                glVertex3fv(vertices[j])
            glEnd()
        
        glPopMatrix()

class EntireCube():
    def __init__(self, N, scale):
        self.N = N
        cr = range(self.N)
        self.cubes = [Cube((x, y, z), self.N, scale) for x in cr for y in cr for z in cr]

    def mainloop(self):
    
        moves_from_start = []
        moves_from_start_down = False
        moves_from_start_key = K_z

        rotate_forward_key = K_SPACE
        rotate_forward_key_down = False
        add_speed_key = K_f
        del_speed_key = K_s
        speed = 20
        
        rot_cube_map  = { K_UP: (-1, 0), K_DOWN: (1, 0), K_LEFT: (0, -1), K_RIGHT: (0, 1)}

        if self.N == 3:
            rot_slice_map = {
                K_1: (0, 0),  K_2: (0, 1),   K_3: (0, 2), 
                K_4: (1, 0),  K_5: (1, 1),   K_6: (1, 2), 
                K_7: (2, 0),  K_8: (2, 1),   K_9: (2, 2),
            }
        elif self.N == 4:
            rot_slice_map = {
                K_1: (0, 0), K_2: (0, 1), K_3: (0, 2), K_4: (0, 3),
                K_5: (1, 0), K_6: (1, 1), K_7: (1, 2), K_8: (1, 3),
                K_q: (2, 0), K_w: (2, 1), K_e: (2, 2), K_r: (2, 3),
            }

        ang_x, ang_y, rot_cube = 0, 0, (0, 0)
        animate, animate_ang, animate_speed = False, 0, speed
        action = (0, 0, 0)
        while True:
            if moves_from_start_down:
                moves_from_start = moves_from_start[::-1]
                for elem in moves_from_start:
                    elem = (elem[0], elem[1], elem[2]*(-1))
                    animate=True
                    while animate:
                        glMatrixMode(GL_MODELVIEW)
                        glLoadIdentity()
                        glTranslatef(0, 0, -40)
                        glRotatef(ang_y, 0, 1, 0)
                        glRotatef(ang_x, 1, 0, 0)

                        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

                        if animate:
                            if animate_ang >= 90:
                                for cube in self.cubes:
                                    cube.update(*elem)
                                animate, animate_ang = False, 0

                        for cube in self.cubes:
                            cube.draw(colors, surfaces, vertices, animate, animate_ang, *elem)
                        if animate:
                            animate_ang += animate_speed
                        pygame.display.flip()
                        pygame.time.wait(10)
                moves_from_start_down = False
                animate=False
                moves_from_start = []

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                if event.type == KEYDOWN:
                    if event.key == moves_from_start_key:
                        moves_from_start_down = True
                        animate = True
                    if event.key == rotate_forward_key:
                        rotate_forward_key_down = True
                    if event.key in [add_speed_key, del_speed_key]:
                        if event.key == add_speed_key and speed < 55:
                            speed += 1
                        elif event.key == del_speed_key and speed > 1:
                            speed -= 1
                        animate_speed = speed
                    if event.key in rot_cube_map:
                        rot_cube = rot_cube_map[event.key]
                    if not animate and event.key in rot_slice_map:
                        animate, action = True, rot_slice_map[event.key][0:2] + tuple([1 if rotate_forward_key_down == True else -1])
                        moves_from_start.append(action)
                        print(f'append {action}')
                if event.type == KEYUP:
                    if event.key == rotate_forward_key:
                        rotate_forward_key_down = False
                    if event.key in rot_cube_map:
                        rot_cube = (0, 0)

            ang_x += rot_cube[0]*2
            ang_y += rot_cube[1]*2

            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            glTranslatef(0, 0, -40)
            glRotatef(ang_y, 0, 1, 0)
            glRotatef(ang_x, 1, 0, 0)

            glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

            if animate:
                if animate_ang >= 90:
                    for cube in self.cubes:
                        cube.update(*action)
                    animate, animate_ang = False, 0

            for cube in self.cubes:
                cube.draw(colors, surfaces, vertices, animate, animate_ang, *action)
            if animate:
                animate_ang += animate_speed

            pygame.display.flip()
            pygame.time.wait(10)

def main():

    pygame.init()
    display = (800,600)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
    glEnable(GL_DEPTH_TEST) 

    glMatrixMode(GL_PROJECTION)
    gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)

    #first parameter is shape of cube!!!
    NewEntireCube = EntireCube(4, 1.5)
    NewEntireCube.mainloop()

if __name__ == '__main__':
    main()
    pygame.quit()
    quit()