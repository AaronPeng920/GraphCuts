import cv2
import numpy as np
from graphmaker import GraphMaker

class CutUI:
    def __init__(self, filename, outfilename):
        self.filename = filename
        self.outfilename = outfilename
        
        self.graph_maker = GraphMaker(filename, outfilename)
        self.display_image = np.array(self.graph_maker.image)
        self.foreground_mode = self.graph_maker.foreground_flag
        
        self.window_name = 'Graph Cut'
        self.started_click = False
    
    def run(self):
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._draw_line_callback)
        
        while 1:
            display = cv2.addWeighted(self.display_image, 0.7, self.graph_maker.get_overlay(), 0.4, 0.3)
            cv2.imshow(self.window_name, display)
            key = cv2.waitKey(20) & 0xFF
            if key == 27:
                break
            # `c` for Clear seeds
            elif key == ord('c'):
                self.graph_maker.clear_seeds()
            # `g` for create Graph
            elif key == ord('g'):
                self.graph_maker.create_graph()
                self.graph_maker.swap_overlay(set_seeds_overlay=False)
            # `t` for Transform mode
            elif key == ord('t'):
                self.foreground_mode = not self.foreground_mode
                self.graph_maker.swap_overlay(set_seeds_overlay=True)
            # `s` for Save
            elif key == ord('s'):
                self.graph_maker.save_image(self.outfilename)
            # `q` for quit
            elif key == ord('q'):
                break
            
        cv2.destroyAllWindows()
    
    
    def _draw_line_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.started_click = True
            self.graph_maker.add_seed(x - 1, y - 1, self.foreground_mode)

        elif event == cv2.EVENT_LBUTTONUP:
            self.started_click = False

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.started_click:
                self.graph_maker.add_seed(x - 1, y - 1, self.foreground_mode)
         