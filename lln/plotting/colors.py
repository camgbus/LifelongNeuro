"""Color mappings for different variables
"""

LIGHT_PALETTE = {'mint': '#9AE2E0', 'light_blue': '#6AB8E8', 'purple': '#A99EEC', 'pink': '#FE8AA8',
                 'orange': '#FFB293', 'yellow': '#F1DD65', 'green': '#ACDD7F', 
                 'mid_gray': '#8A8A8A', 'red': '#EAA4A4'}

CAT_COLORS = dict()
CAT_COLORS['sex'] = {'F': LIGHT_PALETTE['pink'], 'M': LIGHT_PALETTE['light_blue']}
CAT_COLORS['split'] = {'0': LIGHT_PALETTE['mint'], '1': LIGHT_PALETTE['light_blue'],
                       '2': LIGHT_PALETTE['purple'], '3': LIGHT_PALETTE['pink'], 
                       '4': LIGHT_PALETTE['orange']}
