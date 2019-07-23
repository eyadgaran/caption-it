'''
Module for "Home" endpoint
'''

__author__ = 'Elisha Yadgaran'


from quart import render_template


async def home():
    return await render_template('pages/home.html')
