import sys

def error_detail_message(error, error_detail:sys):
    
    '''
    Function to get the error details

    Args:
    error: str: error message
    error_detail: sys: error details

    Returns:
    str: error message with details
    '''

    _, _, tb = error_detail.exc_info()
    file_name = tb.tb_frame.f_code.co_filename
    line_number = tb.tb_lineno
    message = str(error)

    error_message = f'Error occured in {file_name} at line {line_number} with message {message}'

    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):

        '''
        Custom exception class to handle exceptions
        
        Args:
        error_message: str: error message
        error_detail: sys: error details
        '''

        super().__init__(error_message)
        self.error_message = error_detail_message(error_message, error_detail=error_detail)

    def __str__(self):
        '''
        Function to return the error message
        '''
        return self.error_message
    
    
