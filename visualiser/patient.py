from .utils import load_scan, get_pixels_hu


class Patient:
    """Class for 3D CT image of human and his visualisation
    """
    def __init__(self, path: str):
        slices = load_scan(path)
        self.image = get_pixels_hu(slices)

    def horizontal_plot(self, z: int):
        pass
    
    def frontal_plot(self,):
        pass
    
    def longitudinal_plot(self,):
        pass