import sys


def error_message(error, error_detail: sys):
    typ, cls, exc_tb = error_detail.exc_info()
    error_msg = "Error in the filename [{0}] at line [{1}] with " \
                "error message [{2}]".format(exc_tb.tb_frame.f_code.co_filename,exc_tb.tb_lineno,str(error))
    return error_msg

class CustomException(Exception):
    def __init__(self,error_msg,error_detail):
        super.__init__(error_msg)
        self.error_msg = error_message(error_msg,error_detail)

    def __str__(self):
        return self.error_msg
