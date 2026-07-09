css = """
h1 {
    margin-top: 15px;
    text-align: center;
    display:block;
}
.gradio-textbox-updated {
    animation: none !important;
}
.gradio-slider-updated {
    animation: none !important;
}
.scroll-column {
    height: 100px;
    overflow-y: scroll;
}
#start_btn{
    background-color: #0D92F4;
    color: white;
}
.red_btn{
    background-color: #C62E2E;
    color: white;
}
#guide_btn{
    background-color: #495057;
    color: white;
}
.container div.wrap {
    flex-direction: column !important;
}
.gradio-radio {
    flex-direction: column !important;
}
.gradio-checkbox-group{
    flex-direction: column !important;
}
# /* Limit the width of the first AutoEvalColumn so that names don't expand too much */
table td:first-child,
table th:first-child {
    max-width: 400px;
    overflow: auto;
    white-space: nowrap;
}

/* Full width space */
.gradio-container {
    max-width: 85% !important;
}

.download-link {
    display: block;
    margin: 0 auto;
    text-align: center;
    padding: 10px 20px;
    width: 150px;
    background-color: #FFB366; 
    color: white; 
    border: none; 
    border-radius: 8px; 
    cursor: pointer;
    font-size: 16px;
    font-weight: 500;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    transition: all 0.3s ease;
}
"""
