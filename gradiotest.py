#!/usr/bin/env python3

import gradio as gr

def greet(name, intensity): ## Greet takes user's name and number of bangs

    return "Hello, " + name + "!" * int(intensity) ## outputs to the right

demo = gr.Interface( ##create an instance of the gr.Interface class
    ## Interface creates demos, three core arguments
    fn = greet, ## name of function to wrape the UI around
    inputs=["text", "slider"], ## components to use for input = # args in function
    outputs=["text"], ## component to use for output = # of return values
) ## end demo

demo.launch(server_name="127.0.0.1", server_port= 7860)
