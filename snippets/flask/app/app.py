from flask import Flask, render_template, request, redirect, url_for, jsonify

app=Flask(__name__)

@app.before_request
def before_request():
    print("Before the request")

@app.after_request
def after_request():
    print("After the request")
    return response

# base without template
@app.route('/')
def main_page():
    return "hello world"

# render static template
@app.route('/template1')
def index():
    return render_template("index.html")

# render dynamic template
@app.route('/template2')
def dynamic():
    cable_list = ['TDT', 'PDP', 'TKT', 'PKP']
    data = {
        'title': 'Hello Title!',
        'content': 'Welcome to Hello!',
        'cable_list': cable_list,
        'cable_count': len(cable_list)
    }
    return render_template("index2.html", data=data)

# render dynamic template based on a layout
@app.route('/template3')
def dynamic_layout():
    cable_list = ['TDT', 'PDP', 'TKT', 'PKP']
    data = {
        'title': 'Hello Title!',
        'content': 'Welcome to Hello!',
        'cable_list': cable_list,
        'cable_count': len(cable_list)
    }
    return render_template("index3.html", data=data)

# getting data from the user via URL
@app.route('/template4/<cable>/<int:fibres>')
def template4(cable, fibres):
    data = {
        'title': f'Welcome to {cable}',
        'cable': cable,
        'fibres': fibres
    }
    return render_template('template4.html', data=data)

def query_string():
    print(request)
    print(request.args)
    print(request.args.get("cable"))
    return "Ok"

def page_not_found(error):
    #return render_template('404.html'), 404
    
    # another way to handle the error is to use a redirect. It uses the name of the function as an argument
    return redirect(url_for('main_page'))

if __name__ == "__main__":
    app.add_url_rule('/query_string', view_func=query_string)
    app.register_error_handler(404, page_not_found)
    app.run(debug=True, port=5000)