import PySimpleGUI as sg
from main import test, train, evaluate
import threading

# Dark theme color palette
COLORS = {
    'background': '#111111',     # Dark background
    'button_bg': '#2C2C2C',     # Button background
    'button_text': '#FFFFFF',    # White text
    'text': '#FFFFFF',          # White
    'secondary_text': '#8E8E8E'  # Light gray
}

# Updated styling
STYLES = {
    'font_main': ('SF Pro Display', 32),      # Bigger title
    'font_subtitle': ('SF Pro Text', 13),
    'button_size': (12, 1),
    'padding': 20
}

def handle_upload_image():
    filename = sg.popup_get_file('Choose an image file', file_types=(("Image Files", "*.png;*.jpg;*.jpeg"),))
    if filename:
        print(f"Image uploaded: {filename}")
        threading.Thread(target=test, args=(filename,), daemon=True).start()
    return None

def handle_train_model():
    print("Model training initiated")
    # Run train in a separate thread
    threading.Thread(target=train, daemon=True).start()
    
def handle_evaluate_model():
    print("Model evaluation initiated")
    # Run evaluate in a separate thread
    threading.Thread(target=evaluate, daemon=True).start()

def main():
    # Apply the theme
    sg.theme('SystemDefault')
    
    # Custom button styling
    button_style = {
        'size': STYLES['button_size'],
        'font': STYLES['font_subtitle'],
        'button_color': (COLORS['button_text'], COLORS['button_bg']),
        'border_width': 0,
        'pad': ((0, 10), (0, 0)),
        'use_ttk_buttons': False
    }
    
    # Layout with dark theme
    layout = [
        [sg.Text('AI Coin Detector', 
                font=STYLES['font_main'], 
                text_color=COLORS['text'],
                background_color=COLORS['background'],
                pad=((0, 0), (STYLES['padding'], 5)))],
        
        [sg.Text('Detect coins using artificial intelligence', 
                font=STYLES['font_subtitle'],
                text_color=COLORS['secondary_text'],
                background_color=COLORS['background'],
                pad=((0, 0), (0, STYLES['padding']*2)))],
        
        [sg.Button('Upload Image', **button_style),
         sg.Button('Train Model', **button_style),
         sg.Button('Evaluate Model', **button_style)],
         
        [sg.Multiline(size=(60, 15), 
                     font=('Courier', 10),
                     text_color=COLORS['text'],
                     background_color=COLORS['button_bg'],
                     key='-OUTPUT-',
                     disabled=True,
                     autoscroll=True,
                     pad=((0, 0), (STYLES['padding'], 0)))]
    ]

    window = sg.Window('AI Coin Detector', 
                      layout,
                      background_color=COLORS['background'],
                      margins=(STYLES['padding'], STYLES['padding']),
                      finalize=True)

    # Create a queue for thread-safe printing
    import queue
    output_queue = queue.Queue()

    # Redirect stdout to the queue
    import sys
    class OutputRedirector:
        def __init__(self, queue):
            self.queue = queue
            self.stdout = sys.stdout

        def write(self, text):
            self.stdout.write(text)
            self.queue.put(text)

        def flush(self):
            self.stdout.flush()

    sys.stdout = OutputRedirector(output_queue)

    while True:
        event, values = window.read(timeout=100)  # Add timeout for queue checking
        if event == 'OK':
            break
        
        # Process any pending output
        while True:
            try:
                text = output_queue.get_nowait()
                window['-OUTPUT-'].update(text, append=True)
            except queue.Empty:
                break
        
        if event == 'Upload Image':
            handle_upload_image()
        elif event == 'Train Model':
            print('Not ready. Choose another option')
            # handle_train_model()
        elif event == 'Evaluate Model':
            # handle_evaluate_model()
            print('Not ready. Choose another option')
    window.close()

if __name__ == '__main__':
    main()
