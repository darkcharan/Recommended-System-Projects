from movies import get_five_similar_movies
from flask import Flask, request, render_template
from books import get_five_similar_books
import os
from flask import Flask, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
from products import get_five_similar_products
from Food import get_five_similar_food

app = Flask(__name__)

@app.route('/', methods=['GET'])
def search_form():
    choice = request.args.get('choice')

    if choice == 'movie':
        title = request.args.get('name')
        movies = get_five_similar_movies(title)
        if type(movies) == str: 
            return movies
        return render_template('index.html', movies = movies)
    elif choice == 'book':
        title = request.args.get('name')
        books = get_five_similar_books(title)
        if type(books) == str: 
            return books
        return render_template('index.html', books = books)
    elif choice == 'product':
        title = request.args.get('name')
        products = get_five_similar_products(title)
        if type(products) == str: 
            return products
        return render_template('index.html', products = products)
    elif choice == 'food':
        title = request.args.get('name')
        foods = get_five_similar_food(title)
        if type(foods) == str: 
            return foods
        return render_template('index.html', foods = foods)

@app.errorhandler(404) 
def invalid_route(e): 
    return "404 Page not found"


if __name__ == '__main__':
    app.run(debug=True)