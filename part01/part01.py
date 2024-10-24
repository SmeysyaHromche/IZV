#!/usr/bin/env python3
"""
IZV cast1 projektu
Autor: xkukht01

Detailni zadani projektu je v samostatnem projektu e-learningu.
Nezapomente na to, ze python soubory maji dane formatovani.

Muzete pouzit libovolnou vestavenou knihovnu a knihovny predstavene na prednasce
"""
from bs4 import BeautifulSoup
import requests
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from typing import List, Callable, Dict, Any


def distance(a: np.array, b: np.array) -> np.array:
    '''
    Numerics calculations of Euclidean distance
    Parametrs:
        a: [np.array]: input parametrs to math function
        b: [np.array]: input parametrs to math function
    Return:
        [np.array]: Euclidean distance
    '''
    return np.sqrt(np.sum(np.square(a-b), axis=-1))


def generate_graph(a: List[float], show_figure: bool = False, save_path: str | None = None)->None:
    '''
    Generating a graph with different coefficients
    Parametrs:
        a: List[float]: input parametrs to math function
        show_figure: [bool=False]: flag for plots showing
        save_path:   [str | None]: path to saving output of plotting
    Returns:
        None
    '''


    def format_func(value, tick_number)->str:
        '''
        Formating an axis markings to the pi dimension
        Parametrs:
            value: [numbers]: ticks marking
            tick_number: : 
        Return:
            [str]: ticks marking in target format
        '''
        N = int(np.round(2 * value / np.pi))
        if N == 0:
            return '0'
        if N == 2:
            return r'$\pi$'
        elif N % 2 == 0:
            return rf'${{{N//2}}}{{\pi}}$'
        else:
            return rf'$\frac{{{N}}}{{2}}{{\pi}}$' 
    
    
    def func(a:np.array, x:np.array)->np.array:
        '''
        Mathematical implementation of target function
        '''
        a_np = np.array(a).reshape(-1, 1)  # rechape for broadcast
        y = (a_np*a_np)*np.sin(x)
        return y
    

    # math part
    x = np.linspace(0, 6*np.pi, 1000)
    y = func(a, x) 
    
    # prepare display
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot()
    ax.plot()
    ax.set_xlim(0,6*np.pi)
    ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$f_{a}(x)$')
    # plotting
    for i, a_i in enumerate(a):
        ax.plot(x, y[i], label=rf'$y_{{{a_i}}}(x)$')
        ax.fill_between(x, y[i], alpha=0.1)
    # set legend position
    ax.legend(loc='upper center', ncol=len(a), bbox_to_anchor=(0.5, 1.15))
    
    # postprocessing
    if save_path:
        plt.savefig(save_path)
    if show_figure:
        plt.show()
    
    plt.close(fig)


def generate_sinus(show_figure: bool = False, save_path: str | None = None)->None:
    '''
    Task 3: Advanced vizualtization of sin signal
    Parametrs:
        show_figure: [bool=False]: flag for plots showing
        save_path:   [str | None]: path to saving output of plotting
    Return:
        None
    '''

    def func1(t:np.array)->np.array:
        '''
        Implementation of math function by 1
        Params:
        t: [np.array]: input time series
        Returm:
        [np.array]: time series by function
        '''
        y = 0.5*np.cos(np.pi*t/50)
        return y
    
    def func2(t:np.array)->np.array:
        '''
        Implementation of math function by 1
        Params:
        t: [np.array]: input time series
        Returm:
        [np.array]: time series by function
        '''
        sin_sum = np.sin(np.pi*t)+np.sin(3*np.pi*t/2)
        y = 0.25*sin_sum
        return y
    
    # math part
    X_START = 0
    X_STOP = 100
    x = np.linspace(X_START, X_STOP, 10000)
    
    y1 = func1(x)
    y2 = func2(x)
    y3 = y1+y2
    y_base = [y1, y2]

    # masks for separate different area of third function graph
    y3_not_green = np.ma.masked_where(y3 > y1, y3)
    y3_green = np.ma.masked_where(y3 <= y1, y3)

    # display
    fig, axs = plt.subplots(3, 1, figsize=(10,6))
    for i, ax in enumerate(axs):
        ax.set_xlim(0, 100)
        ax.xaxis.set_major_locator(plt.MultipleLocator(25))
        ax.set_ylim(-0.8, 0.8)
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.4))
        if i < 2:
            ax.set_ylabel(rf'$f_{{{i+1}}}(t)$')
            ax.tick_params(axis='x', labelbottom=False, labeltop=False)
            ax.plot(x, y_base[i])
        else:
            ax.set_ylabel(r'$f_{1}(t) + f_{2}(t)$')
            ax.tick_params(axis='x', labelbottom=True, labeltop=False)
            ax.plot(x, y3_green, color='green')
            ax.plot(x[:len(x)//2],y3_not_green[:len(x)//2], color='red')
            ax.plot(x[len(x)//2:], y3_not_green[len(x)//2:], color='darkorange')
    
    # postprocessing
    if save_path:
        plt.savefig(save_path)
    if show_figure:
        plt.show()
    
    plt.close(fig)


def download_data() -> Dict[str, List[Any]]:
    '''
    Downloading table from web
    Return:
        Dict[str, List[Any]]: table form web after formatting
    '''
    # convert string with geo coordination to float
    coord_to_float = lambda x: float(x[:-1].replace(',', '.'))
    # remove white space from string with float number and convert it to float
    nums_without_wh_sp = lambda x: float(''.join(ch for ch in x if ch.isdigit() or ch in ',.').replace(',', '.'))
    
    # target structure
    out = {'positions':[], 'lats': [], 'longs': [], 'heights':[]}
    
    # endpoint
    URL_TABLE = 'https://ehw.fit.vutbr.cz/izv/st_zemepis_cz.html'
    
    # download page
    response = requests.get(URL_TABLE)
    if response.status_code != 200:
        return dict()
    
    # transform to issie parsed structure
    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find_all('table')[-1].find_all('tr')[1:] # last table
    # parsing
    for line in table:
        subinfo = line.find_all('td')[::2]
        out['positions'].append( subinfo[0].find('strong').text)
        out['lats'].append(coord_to_float(subinfo[1].text))
        out['longs'].append(coord_to_float(subinfo[2].text))
        out['heights'].append(nums_without_wh_sp(subinfo[3].text))
    
    return out


if __name__ == "main":
    generate_graph([7, 4, 3])
    generate_sinus()
