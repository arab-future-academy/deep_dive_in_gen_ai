class AFACalc:
    CATEGORY = "AFA"
    @classmethod    
    def INPUT_TYPES(s):
        return { "required":  {
            "a": ("INT,FLOAT", {
                "default": "1"
            }),
            "op": (["+", "-", "*", "/"], {
                "default": "+"
            }),
            "b": ("INT,FLOAT", {
                "default": "1"
            }),
        }}
    RETURN_TYPES = ("INT", "FLOAT")
    RETURN_NAMES = ("OUT INT", "OUT FLOAT")
    FUNCTION = "do_calc"
    
    def __init__(self):
        super().__init__()

    def do_calc(self, a, op, b):
        res = 0
        if op == "+":
            res = a + b
        elif op == "-":
            res = a - b
        elif op == "*":
            res = a * b
        elif op == "/":
            res = a/b
        return (int(res), float(res))
    

