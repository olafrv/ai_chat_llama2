import gradio as gr

def greet(name):
    return "Hello " + name + "!"

app = gr.Interface(fn=greet, inputs="text", outputs="text")
    
app.launch() 