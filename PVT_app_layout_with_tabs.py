# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 12:41:41 2021

@author: Daniel E. Crosby
"""


import json

import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.PULSE])

# Define functions used for the PVT calculations
def calc_P_bp(T, R_sb, rho_o_sc, rho_g_sc):
    help01 = (10**(0.00164*T))/(10**(1768/rho_o_sc))
    Pbp = 125e3 * ((716*R_sb/rho_g_sc)**0.83 * help01 - 1.4)
    return Pbp

def calc_dead_oil_visc(rho_o_sc, T):
    b = 5.693 - (2.863e3/rho_o_sc)
    a = (10**b)/((1.8*T + 32)**1.163)
    u_od = 1e-3*(10**a - 1)
    return u_od

def calc_bubbl_pt_oil_visc(R_sb, u_od):
    c = 5.44*(R_sb/.178 + 150)**-0.338
    u_ob = ((10.72*1e-3)*((R_sb/0.178 + 100)**-0.515))*((u_od*1e3)**c)
    return u_ob

def calc_Rs_below_bp(rho_o_sc, rho_g_sc, T, p):
    help02 = 10**(1768/rho_o_sc - 0.00164*T)
    Rs = (rho_g_sc/716)*((8e-6*p+1.4)*help02)**1.2048
    return Rs

def calc_Bo_below_bp(rho_g_sc, rho_o_sc, Rs, T):
    help03 = np.sqrt(rho_g_sc/rho_o_sc)
    B_o = 0.9759 + 12e-5 *(160 * Rs * help03 + 2.25 * T + 40)**1.2
    return B_o

def calc_Uo_below_bp(Rs, u_od):
    c = 5.44*(Rs/.178 + 150)**-0.338
    u_o = ((10.72*1e-3)*((Rs/0.178 + 100)**-0.515))*((u_od*1e3)**c)
    return u_o

def calc_Bo_bp(rho_g_sc, rho_o_sc, Rs, T):
    help03 = np.sqrt(rho_g_sc/rho_o_sc)
    Bo_bp = 0.9759 + 12e-5*(160*Rs*help03+2.25*T+40)**1.2
    return Bo_bp

def calc_rho_o_below_bp(rho_o_sc, Rs, rho_g_sc, Bo):
    rho_o_below_bp = (rho_o_sc + Rs*rho_g_sc)/Bo
    return rho_o_below_bp

def calc_rho_o_at_bp(rho_o_sc, Rsb, rho_g_sc, Bo_bp):
    rho_o_bp = (rho_o_sc + Rsb*rho_g_sc)/Bo_bp
    return rho_o_bp

def calc_rho_o_above_bp(rho_o_bp, c_o, P, Pbp):
    rho_o_above_bp = rho_o_bp*np.exp(c_o*(P - Pbp))
    return rho_o_above_bp

def Z_factor_Papay(p_pr,T_pr):
    Z = 1 - 3.52*p_pr/(T_pr*10**0.9813) + 0.274*p_pr**2/(T_pr*10**0.8157)
    return Z

def calc_Z_factor(p_pr, T_pr):
    # Coefficients:
    a1 =  0.3265
    a2 = -1.0700
    a3 = -0.5339
    a4 =  0.01569
    a5 = -0.05165
    a6 =  0.5475
    a7 = -0.7361
    a8 =  0.1844
    a9 =  0.1056
    a10 = 0.6134
    a11 = 0.7210

    c = 0.27 * p_pr/T_pr

    b1 = c * (a1 + a2/T_pr + a3/T_pr**3 + a4/T_pr**4 + a5/T_pr**5)
    b2 = c**2 * (a6 + a7/T_pr + a8/T_pr**2)
    b3 = c**5 * a9*(a7/T_pr + a8/T_pr**2)
    b4 = c**2 * a10/T_pr**3
    b5 = c**2 * a11
    b6 = b4 * b5

    # Initiate Z with the Papay correlation:
    Z_0 = Z_factor_Papay(p_pr,T_pr)
    Z = Z_0
    
    # Improve the result with Newton Raphson iteration:
    tol_abs = 1.e-6 # Absolute convergence criterion
    tol_rel = 1.e-9 # Relative convergence criterion
    max_diff = 0.3 # Maximum allowed absolute difference in Z per iteration step
    itern = 0 # Iteration counter
    repeat = 1

    while repeat > 0:       
        itern = itern+1
        Z_old = Z
        help01 = Z_old - b1*Z_old**-1 - b2*Z_old**-2 + b3*Z_old**-5
        help02 = -(b4*Z_old**-2 + b6*Z_old**-4) * np.exp(-b5*Z_old**-2) - 1
        fZ = help01 + help02

        help03 = 1 + b1*Z_old**-2 + 2*b2*Z_old**-3 - 5*b3*Z_old**-6
        help04 = (2*b4*Z_old**-3 - 2*b4*b5*Z_old**-5 + 4*b6*Z_old**-5 - 2*b5*b6*Z_old**-7)* np.exp(-b5*Z_old**-2)
        dfZdZ = help03 + help04

        Z = Z_old - fZ/dfZdZ # Newton Raphson iteration
        diff = Z-Z_old

        if abs(diff) > max_diff: # Check if steps are too large
            Z = Z_old + max_diff * np.sign(diff) # Newton Raphson iteration with reduced step size
            diff = max_diff

        rel_diff = diff/Z_old

        if abs(diff) > tol_abs: # Check for convergence
            repeat = 1
        else:
            if abs(rel_diff) > tol_rel:
                repeat = 1
            else:
                repeat = 0
    return Z

def calc_Bg(T, P, Z):
    T_abs = T + 273.15
    Tsc_abs = 60 + 273.15
    Bg = 100e3*T_abs*Z/(P*Tsc_abs)
    return Bg

# Define the range of temperatures and pressures used for the PVT calculations:
temperatures = np.arange(30, 110, 10).tolist()
pressures = np.linspace(40e3, 30e6, 40).tolist()
# Define the PVT parameters available to plot in the sensitivity chart
PVT_parameters = ['Bo', 'Uo', 'Rs', 'Rho_o']
# Define the table and header data
table_header = [
    html.Thead(html.Tr([html.Th("Pbp, MPa"), html.Th("Bo @ Pbp, rm3/m3"),
                        html.Th("Uo @ Pbp, Pa-s"), html.Th("Co @ Pbp, 1/Pa"), html.Th("Rho_oil @ bp, kg/m3")]))
]

row1 = html.Tr([html.Td(id="P_bp"), html.Td(id="Bo_bp"), html.Td(id="u_ob"), html.Td(id="c_o"), html.Td(id="rho_o_bp")])

table_body = [html.Tbody([row1])]

table = dbc.Table(table_header + table_body, bordered=True, striped=True,
    style={'overflowX': 'auto',
          'background-color': 'white',
          'border-color':'black',
          'margin-left':'2vw',
          'margin-right':'2vw',
          'text-align': 'center',
          'width':'85%'}
    )
# Define the app layout
app.layout = html.Div([
    dcc.Tabs(id="tabs", className='custom-tabs-container', children=[
            dcc.Tab(label='PVT Calculator', className='custom-tab',
                selected_className='custom-tab--selected', children=[
                    dbc.Row(
                                [
                                    dbc.Col(html.H4(["Input PVT Data:", dbc.Badge("SI units", className="ml-1")]))
                                ],
                                justify="center", style={'margin-bottom':'2vw', 'margin-left':'2vw'}
                            ),
            
                    dbc.Row(
                        dbc.Col(
                            html.Div([
                                    dcc.Input(
                                    id='R_sb',
                                    type='number',
                                    placeholder='insert R_sb',
                                    debounce=True,
                                    required=True,
                                    size='10'
                                ),
                                dcc.Input(
                                    id='rho_g_sc',
                                    type='number',
                                    placeholder='insert rho_g_sc',
                                    debounce=True,
                                    required=True,
                                    size='10'
                                ),
                                dcc.Input(
                                    id='rho_o_sc',
                                    type='number',
                                    placeholder='insert rho_o_sc',
                                    debounce=True,
                                    required=True,
                                    size='10'
                                ),
                                dcc.Input(
                                    id='T',
                                    type='number',
                                    placeholder='insert T, deg. C',
                                    debounce=True,
                                    required=True,
                                    size='10'
                                ),
                                dcc.Input(
                                    id='P_sep',
                                    type='number',
                                    placeholder='insert Psep, Pa',
                                    debounce=True,
                                    required=True,
                                    size='10'
                                    )
                                ]), width=5, lg={'size':9, 'offset':1, 'order':'first'}, style={'margin-bottom':'2vw',
                                                                                                'display': 'inline-block'}
                            )
                        ),
                    dbc.Row(
                                [
                                    dbc.Col(html.H4(["Calculated PVT Data:", dbc.Badge("SI units", className='ml-1')])),
                                ],
                                justify="center", style={'margin-bottom':'2vw', 'margin-left':'2vw'}
                            ),
                    dbc.Row(
                            dbc.Col(html.Div(table))
                        ),
                    dbc.Row(
                        [
                            dbc.Col(dcc.Graph(id='Rs_chart'),
                                    width=5, lg={'size':5, 'offset':0, 'order':'first'},
                                    style={'background-color': '#FEFEFE','box-shadow': '8px 8px 8px #A67CF3',
                                           'border-radius': '15px 15px 15px 15px', 'padding': '6px 6px 6px 6px',
                                           'float': 'left', 'display': 'inline-block', 'margin-left':'2vw'}
                                    ),
                            dbc.Col(dcc.Graph(id='Bo_chart'),
                                    width=5, lg={'size':5, 'offset':0, 'order':'last'},
                                    style={'background-color': '#FEFEFE','box-shadow': '8px 8px 8px #A67CF3',
                                           'border-radius': '15px 15px 15px 15px','padding': '6px 6px 6px 6px',
                                           'float': 'right', 'display': 'inline-block', 'margin-left':'2vw'}
                                    )
                            ]),
                    dbc.Row(
                        [
                            dbc.Col(dcc.Graph(id='Uo_chart'),
                                    width=5, lg={'size':0, 'offset':0, 'order':'first'},
                                    style={'background-color': '#FEFEFE','box-shadow': '8px 8px 8px #A67CF3',
                                           'border-radius': '15px 15px 15px 15px','padding': '6px 6px 6px 6px',
                                           'float': 'left', 'display': 'inline-block', 'margin-left':'2vw',
                                           'margin-top':'2vw'}
                                    ),
                            dbc.Col(dcc.Graph(id='Bg_chart'),
                                    width=5, lg={'size':0, 'offset':0, 'order':'last'},
                                    style={'background-color': '#FEFEFE','box-shadow': '8px 8px 8px #A67CF3',
                                           'border-radius': '15px 15px 15px 15px','padding': '6px 6px 6px 6px',
                                           'float': 'right', 'display': 'inline-block', 'margin-left':'2vw',
                                           'margin-top':'2vw'}
                                    )
                            ]
                        ),
                     dbc.Row(
                         dbc.Col(dcc.Graph(id='rho_o_chart'),
                            width=5, lg={'size':0, 'offset':0, 'order':'last'},
                                    style={'background-color': '#FEFEFE','box-shadow': '8px 8px 8px #A67CF3',
                                           'border-radius': '15px 15px 15px 15px','padding': '6px 6px 6px 6px',
                                           'float': 'right', 'display': 'inline-block', 'margin-left':'2vw',
                                           'margin-top':'2vw'}
                                    )
                         ),
                    ]),
            dcc.Tab(label='PVT Temperature Sensitivity Analysis', className='custom-tab',
                selected_className='custom-tab--selected', children=[
                    dbc.Row(
                                [
                                    dbc.Col(html.H4(["Input basic PVT data:"]))
                                ],
                                justify="center", style={'margin-bottom':'1vw', 'margin-left':'2vw'}
                            ),
                    dbc.Row(
                        dbc.Col(
                            html.Div([
                                dcc.Input(
                                id='R_sb2',
                                type='number',
                                placeholder='insert R_sb',
                                debounce=True,
                                required=True,
                                size='15'
                            ),
                            dcc.Input(
                                id='rho_g_sc2',
                                type='number',
                                placeholder='insert rho_g_sc',
                                debounce=True,
                                required=True,
                                size='15'
                            ),
                            dcc.Input(
                                id='rho_o_sc2',
                                type='number',
                                placeholder='insert rho_o_sc',
                                debounce=True,
                                required=True,
                                size='15'
                                ),
                            dcc.Input(
                                id='P_sep2',
                                type='number',
                                placeholder='insert Psep, Pa',
                                debounce=True,
                                required=True,
                                size='15'
                                ),
                            ]),width=6, lg={'size':9, 'offset':1, 'order':'first'}, style={'margin-bottom':'2vw',
                                                                                                'display': 'inline-block'}
                        ),
                    ),
                    dbc.Row(
                        dbc.Col(
                            dcc.Dropdown(
                                id='PVT_parameter',
                                options=[ {'label': i, 'value': i} for i in PVT_parameters],
                                value='Bo'
                            ),
                            width=3, lg={'size':3, 'offset':5}, 
                                style={'background-color': '#FEFEFE','box-shadow': '8px 8px 8px #A67CF3',
                                        'border-radius': '15px 15px 15px 15px','padding': '6px 6px 6px 6px',
                                        'margin-top':'2vw', 'font-size' : '75%'}
                        )
                    ),
                    dbc.Row(
                        [
                        dbc.Col(dcc.Graph(id='Pbp_chart'),
                            width=5, lg={'size':5, 'offset':0, 'order':'first'},
                            style={'background-color': '#FEFEFE','box-shadow': '8px 8px 8px #A67CF3',
                                    'border-radius': '15px 15px 15px 15px','padding': '6px 6px 6px 6px',
                                    'float': 'right', 'display': 'inline-block', 'margin-left':'2vw',
                                    'margin-top':'2vw'}
                        ),
                        dbc.Col(dcc.Graph(id='Bo_sensitivity_chart'),
                                width=5, lg={'size':5, 'offset':0, 'order':'last'},
                                style={'background-color': '#FEFEFE','box-shadow': '8px 8px 8px #A67CF3',
                                        'border-radius': '15px 15px 15px 15px','padding': '6px 6px 6px 6px',
                                        'float': 'right', 'display': 'inline-block', 'margin-left':'2vw',
                                        'margin-top':'2vw'}
                                    )
                        ]
                    ),
                        dbc.Col(
                            dcc.Slider(
                                id='selected_temp',
                                min=30.0,
                                max=100.0,
                                value=60.0,
                                marks={
                                    30: {'label': '30 °C', 'style': {'color': '#77b0b1'}},
                                    40: {'label': '40 °C'},
                                    50: {'label': '50 °C'},
                                    60: {'label': '60 °C'},
                                    70: {'label': '70 °C'},
                                    80: {'label': '80 °C'},
                                    90: {'label': '90 °C'},
                                    100: {'label': '100 °C', 'style': {'color': '#f50'}}
                                }
                            ),
                             width=6, lg={'size':6, 'offset':5}, 
                                style={'background-color': '#FEFEFE','box-shadow': '8px 8px 8px #A67CF3',
                                        'border-radius': '15px 15px 15px 15px','padding': '6px 6px 6px 6px',
                                        'margin-top':'2vw'}
                        )
                ]),
            ]),
        ])

@app.callback(
    Output('Rs_chart', 'figure'),
    Output('P_bp', 'children'),
    Output('Bo_chart', 'figure'),
    Output('Bo_bp', 'children'),
    Output('Uo_chart', 'figure'),
    Output('Bg_chart', 'figure'),
    Output('u_ob', 'children'),
    Output('c_o', 'children'),
    Output('rho_o_chart', 'figure'),
    Output('rho_o_bp', 'children'),
    [Input('R_sb', 'value'),
    Input('rho_g_sc', 'value'),
    Input('rho_o_sc', 'value'),
    Input('T', 'value'),
    Input('P_sep', 'value')  
    ])
def Update_PVT_charts(R_sb, rho_g_sc, rho_o_sc, T, P_sep):
    Rs_data = []
    Bo_data = []
    Uo_data = []
    rho_o_data = []
    Bg_data = []
    # Calculate the bubble point pressure:
    Pbp = calc_P_bp(T, R_sb, rho_o_sc, rho_g_sc)
    # Calculate the dead oil viscosity:
    u_od = calc_dead_oil_visc(rho_o_sc, T)
    # Calculate the bubble point viscosity:
    u_ob = calc_bubbl_pt_oil_visc(R_sb, u_od)
    for p in pressures:
        if p <= Pbp:
            Rs = calc_Rs_below_bp(rho_o_sc, rho_g_sc, T, p)
            B_o = calc_Bo_below_bp(rho_g_sc, rho_o_sc, Rs, T)
            u_o = calc_Uo_below_bp(Rs, u_od)
            rho_o = calc_rho_o_below_bp(rho_o_sc, Rs, rho_g_sc, B_o)
            Ppc = 5218*1e3 - 734*1e3*rho_g_sc - 16.4*1e3*rho_g_sc**2
            Tpc = 94 + 157.9*rho_g_sc - 27.2*rho_g_sc**2
            p_pr = p/Ppc
            T_pr = (T+273.15)/Tpc
            Z = calc_Z_factor(p_pr, T_pr)
            Bg = calc_Bg(T, p, Z)

        else:
            Rs=R_sb
            Bo_bp = calc_Bo_bp(rho_g_sc, rho_o_sc, Rs, T)
            rho_o_bp = calc_rho_o_at_bp(rho_o_sc, Rs, rho_g_sc, Bo_bp)
            Bg = 0
            # Compute the oil compressibility above the bubble point pressure:
            term_1 = (5.912e-5*((141.5e3/rho_o_sc)-131.5))*(1.8*T + 32)*np.log10(P_sep/790.8e3)
            rho_g_100 = rho_g_sc*(1+term_1)
            c_o = (-2541 + (27.8*R_sb) + (31*T) - (959*rho_g_100) + (1784e3/rho_o_sc))/(100000*p)
            B_o = Bo_bp*np.exp(-c_o*(p-Pbp))
            d = (7.2*1e-5)*(p**1.187)*np.exp(-11.513 - 1.30*1e-8*p)
            u_o = u_ob*(p/Pbp)**d
            # Compute the oil density above the bubble point pressure
            rho_o = calc_rho_o_above_bp(rho_o_bp, c_o, p, Pbp)
        Rs_data.append(Rs)
        Bo_data.append(B_o)
        Uo_data.append(u_o)
        Bg_data.append(Bg)
        rho_o_data.append(rho_o)

    df = pd.DataFrame({
    "x": pressures,
    "y": Rs_data,
    "y2": Bo_data,
    "y3": Uo_data,
    "y4": Bg_data,
    'y5': rho_o_data
    })

    fig1 = px.line(df, x="x", y="y", title='Solution GOR vs Pressure',
        labels={
                "x": "Pressure, Pa",
                "y": "Solution GOR, m<sup>3</sup>/m<sup>3</sup>"
        })
    fig1.update_traces(mode='lines+markers', line=dict(color='DarkRed'), marker=dict(opacity=0.5))
    fig1.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='LightGrey', zeroline=True, zerolinewidth=1, zerolinecolor='LightGrey')
    fig1.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='LightGrey', zeroline=True, zerolinewidth=1, zerolinecolor='LightGrey')
    fig1.update_layout(
        font_family="Georgia", paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)')

    fig2 = px.line(df, x="x", y="y2", title='Oil Formation Volume Factor vs Pressure',
        labels={
                "x": "Pressure, Pa",
                "y2": "Bo, rm<sup>3</sup>/m<sup>3</sup>"
        })
    fig2.update_traces(mode='lines+markers', line=dict(color='Blue'), marker=dict(opacity=0.5))
    fig2.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='LightGrey', zeroline=True, zerolinewidth=1, zerolinecolor='LightGrey')
    fig2.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='LightGrey', zeroline=True, zerolinewidth=1, zerolinecolor='LightGrey')    
    fig2.update_layout(
        font_family="Georgia", paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)')

    fig3 = px.line(df, x="x", y="y3", title='Oil Viscosity vs Pressure',
        labels={
                "x": "Pressure, Pa",
                "y3": "Uo, Pa-s"
        })
    fig3.update_traces(mode='lines+markers', line=dict(color='#BA4A00'), marker=dict(opacity=0.5))
    fig3.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='LightGrey', zeroline=True, zerolinewidth=1, zerolinecolor='LightGrey')
    fig3.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='LightGrey', zeroline=True, zerolinewidth=1, zerolinecolor='LightGrey')
    fig3.update_layout(
        font_family="Georgia", paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)')


    fig4 = px.line(df, x="x", y="y4", title='Gas Formation Volume Factor vs Pressure',
        labels={
                "x": "Pressure, Pa",
                "y4": "Bg, m<sup>3</sup>/m<sup>3</sup>"
        })
    fig4.update_traces(mode='lines+markers', line=dict(color='#FA52DE'), marker=dict(opacity=0.5))
    fig4.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='LightGrey', zeroline=True, zerolinewidth=1, zerolinecolor='LightGrey')
    fig4.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='LightGrey', zeroline=True, zerolinewidth=1, zerolinecolor='LightGrey')
    fig4.update_layout(
        font_family="Georgia", paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)')

    fig5 = px.line(df, x="x", y="y5", title='Oil Density vs Pressure',
        labels={
                "x": "Pressure, Pa",
                "y5": "Oil Density, kg/m<sup>3</sup>"
        })
    fig5.update_traces(mode='lines+markers', line=dict(color='Green'), marker=dict(opacity=0.5))
    fig5.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='LightGrey', zeroline=True, zerolinewidth=1, zerolinecolor='LightGrey')
    fig5.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='LightGrey', zeroline=True, zerolinewidth=1, zerolinecolor='LightGrey')
    fig5.update_layout(
        font_family="Georgia", paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)')

    return fig1, round(Pbp/1e6, 2), fig2, round(Bo_bp, 2), fig3, fig4, round(u_ob, 4), "{:e}".format(c_o), fig5, round(rho_o_bp, 2)

@app.callback(
    Output('Pbp_chart', 'figure'),
    [Input('R_sb2', 'value'),
    Input('rho_g_sc2', 'value'),
    Input('rho_o_sc2', 'value')
    ])

def Pbp_sensitivity_chart(R_sb2, rho_g_sc2, rho_o_sc2):
    Pbp_data = []
    
    for t in temperatures:
        Pbp2 = calc_P_bp(t, R_sb2, rho_o_sc2, rho_g_sc2)
        Pbp_data.append(Pbp2)

    df2 = pd.DataFrame({
    "x": temperatures,
    "y": Pbp_data,
    })

    fig = px.scatter(df2, x='x', y='y',
        labels={
                "x": "Temperature, °C",
                "y": "Bubble Point Pressure, Mpa"
            })
    fig.update_traces(mode='lines+markers', line=dict(color='Violet'), marker=dict(color='Violet', opacity=0.5))
    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='LightGrey', zeroline=True, zerolinewidth=1, zerolinecolor='LightGrey')
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='LightGrey', zeroline=True, zerolinewidth=1, zerolinecolor='LightGrey')
    fig.update_layout(title='Bubble Point Pressure', font_family="Georgia", paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)')

    return fig

@app.callback(
    Output('Bo_sensitivity_chart', 'figure'),
    [Input('R_sb2', 'value'),
    Input('rho_g_sc2', 'value'),
    Input('rho_o_sc2', 'value'),
    Input('P_sep2', 'value'),
    Input('PVT_parameter', 'value'),
    Input('selected_temp', 'value')
    ])

def Pbp_sensitivity_chart(R_sb2, rho_g_sc2, rho_o_sc2, P_sep2, y_axis_column_name, selected_temp):
    Bo_sens_data = []
    Rs_sens_data = []
    Uo_sens_data = []
    Rho_o_sens_data = []
    #T2 = selected_temp
    Pbp3 = calc_P_bp(selected_temp, R_sb2, rho_o_sc2, rho_g_sc2)
    # Calculate the dead oil viscosity:
    u_od2 = calc_dead_oil_visc(rho_o_sc2, selected_temp)
    # Calculate the bubble point viscosity:
    u_ob2 = calc_bubbl_pt_oil_visc(R_sb2, u_od2)    

    for p in pressures:
        if p <= Pbp3:
            Rs2 = calc_Rs_below_bp(rho_o_sc2, rho_g_sc2, selected_temp, p)
            B_o2 = calc_Bo_below_bp(rho_g_sc2, rho_o_sc2, Rs2, selected_temp)
            u_o2 = calc_Uo_below_bp(Rs2, u_od2)
            rho_o2 = calc_rho_o_below_bp(rho_o_sc2, Rs2, rho_g_sc2, B_o2)
        else:
            Rs2=R_sb2
            Bo_bp2 = calc_Bo_bp(rho_g_sc2, rho_o_sc2, Rs2, selected_temp)
            rho_o_bp2 = calc_rho_o_at_bp(rho_o_sc2, Rs2, rho_g_sc2, Bo_bp2)
            # Compute the oil compressibility above the bubble point pressure:
            term_2 = (5.912e-5*((141.5e3/rho_o_sc2)-131.5))*(1.8*selected_temp + 32)*np.log10(P_sep2/790.8e3)
            rho_g_100 = rho_g_sc2*(1+term_2)
            c_o = (-2541 + (27.8*R_sb2) + (31*selected_temp) - (959*rho_g_100) + (1784e3/rho_o_sc2))/(100000*p)
            B_o2 = Bo_bp2*np.exp(-c_o*(p-Pbp3))
            d2 = (7.2*1e-5)*(p**1.187)*np.exp(-11.513 - 1.30*1e-8*p)
            u_o2 = u_ob2*(p/Pbp3)**d2
            # Compute the oil density above the bubble point pressure
            rho_o2 = calc_rho_o_above_bp(rho_o_bp2, c_o, p, Pbp3)

        Bo_sens_data.append(B_o2)
        Uo_sens_data.append(u_o2)
        Rs_sens_data.append(Rs2)
        Rho_o_sens_data.append(rho_o2)

    df3 = pd.DataFrame({
        "Pressure": pressures,
        "Bo": Bo_sens_data,
        "Uo": Uo_sens_data,
        "Rs": Rs_sens_data,
        "Rho_o": Rho_o_sens_data
        })

    fig = px.scatter(df3, x="Pressure", y=y_axis_column_name,
        labels={
                    "Pressure": "Pressure, MPa",
                    "y": y_axis_column_name
                })
    fig.update_traces(mode='lines+markers', line=dict(color='Blue'), marker=dict(opacity=0.5))
    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='LightGrey', zeroline=True, zerolinewidth=1, zerolinecolor='LightGrey')
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='LightGrey', zeroline=True, zerolinewidth=1, zerolinecolor='LightGrey')
    fig.update_layout(title= y_axis_column_name + ' Sensitivity to Temperature and Pressure', font_family="Georgia", 
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')    

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)


