import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation, rc
from IPython.display import HTML
from matplotlib.patches import Arc

def circarrow(axis, diameter, centX, centY, startangle, angle,**kwargs):
    startarrow=kwargs.pop("startarrow",False)
    endarrow=kwargs.pop("endarrow",False)

    arc = Arc([centX,centY],diameter,diameter,angle=startangle,
          theta1=np.rad2deg(kwargs.get("head_length",1.5*3*.001)) if startarrow else 0,theta2=angle-(np.rad2deg(kwargs.get("head_length",1.5*3*.001)) if endarrow else 0),linestyle="-",color=kwargs.get("color","black"))
    axis.add_patch(arc)

    start_arrow = None
    end_arrow = None
    if startarrow:
        startX=diameter/2*np.cos(np.radians(startangle))
        startY=diameter/2*np.sin(np.radians(startangle))
        startDX=+.000001*diameter/2*np.sin(np.radians(startangle)+kwargs.get("head_length",1.5*3*.001))
        startDY=-.000001*diameter/2*np.cos(np.radians(startangle)+kwargs.get("head_length",1.5*3*.001))
        start_arrow = axis.arrow(centX+startX-startDX,centY+startY-startDY,startDX,startDY,**kwargs)

    if endarrow:
        endX=diameter/2*np.cos(np.radians(startangle+angle))
        endY=diameter/2*np.sin(np.radians(startangle+angle))
        endDX=-.000001*diameter/2*np.sin(np.radians(startangle+angle)-kwargs.get("head_length",1.5*3*.001))
        endDY=+.000001*diameter/2*np.cos(np.radians(startangle+angle)-kwargs.get("head_length",1.5*3*.001))
        end_arrow = axis.arrow(centX+endX-endDX,centY+endY-endDY,endDX,endDY,**kwargs)
    
    return arc, start_arrow, end_arrow

def update_circarrow(axis, arc, start_arrow, end_arrow, 
                 diameter, centX, centY, startangle, angle, **kwargs):
    startarrow=kwargs.pop("startarrow",False)
    endarrow=kwargs.pop("endarrow",False)

    arc.center = [centX, centY]
    arc.width = diameter
    arc.height = diameter
    arc.angle = startangle
    arc.theta1 = np.rad2deg(kwargs.get("head_length",1.5*3*.001)) if startarrow else 0
    arc.theta2 = angle-(np.rad2deg(kwargs.get("head_length",1.5*3*.001)) if endarrow else 0)

    if startarrow:
        startX=diameter/2*np.cos(np.radians(startangle))
        startY=diameter/2*np.sin(np.radians(startangle))
        startDX=+.000001*diameter/2*np.sin(np.radians(startangle)+kwargs.get("head_length",1.5*3*.001))
        startDY=-.000001*diameter/2*np.cos(np.radians(startangle)+kwargs.get("head_length",1.5*3*.001))
        if start_arrow is not None:
            start_arrow.remove()
        start_arrow = axis.arrow(centX+startX-startDX, centY+startY-startDY, startDX, startDY, **kwargs)
    elif start_arrow is not None:
        start_arrow.remove()
        start_arrow = None

    if endarrow:
        endX=diameter/2*np.cos(np.radians(startangle+angle))
        endY=diameter/2*np.sin(np.radians(startangle+angle))
        endDX=-.000001*diameter/2*np.sin(np.radians(startangle+angle)-kwargs.get("head_length",1.5*3*.001))
        endDY=+.000001*diameter/2*np.cos(np.radians(startangle+angle)-kwargs.get("head_length",1.5*3*.001))
        if end_arrow is not None:
            # end_arrow.x = centX+endX-endDX
            # end_arrow.y = centY+endY-endDY
            # end_arrow.dx = endDX
            # end_arrow.dy = endDY
            end_arrow.remove()
        end_arrow = axis.arrow(centX+endX-endDX,centY+endY-endDY,endDX,endDY,**kwargs)
    elif end_arrow is not None:
        end_arrow.remove()
        end_arrow = None
    
    return arc, start_arrow, end_arrow