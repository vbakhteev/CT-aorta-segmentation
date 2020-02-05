


class Patient:
    """Class for 3D CT image of human and his visualisation
    """
    def __init__(self, path: str):
        slices = load_scan(path)
        self.image = get_pixels_hu(slices)

    def horizontal_plot(self, z: int, hu_bounds:=(0, 60), hu_type: HU_Type=None):
        if hu_range:
            hu_bounds = hu_type_to_bounds[hu_range]
        
        raise NotImplementedError('')
    
    def frontal_plot(self,):
        pass
    
    def longitudinal_plot(self,):
        pass