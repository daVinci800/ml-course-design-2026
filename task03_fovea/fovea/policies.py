"""Fixed inference candidates; choose using held-out images, never test GT."""
import math

POLICIES=('integral_fixed','dark_fixed','integral_adaptive','dark_adaptive')


def policy_settings(settings,policy):
    result=dict(settings)
    if policy is None or policy=='checkpoint':
        return result
    if policy not in POLICIES:
        raise ValueError(f'Unknown inference policy: {policy}')
    decoder,roi=policy.split('_')
    result['decoder']=decoder
    result['adaptive_crop']=roi=='adaptive'
    result.setdefault('adaptive_crop_max',768.0)
    result.setdefault('adaptive_crop_spread_factor',4.0)
    return result


def refinement_windows(predictions,settings):
    base=float(settings['crop_side'])
    maximum=float(settings.get('adaptive_crop_max',base))
    factor=float(settings.get('adaptive_crop_spread_factor',4.0))
    adaptive=settings.get('adaptive_crop',False)
    if base<=0 or maximum<base or factor<0 or not all(map(math.isfinite,[base,maximum,factor])):
        raise ValueError('Invalid refinement crop parameters.')
    centers={i:p['xy'] for i,p in predictions.items()}
    sides={}
    for ident,p in predictions.items():
        spread=float(p.get('tta_spread',0))
        if not math.isfinite(spread) or spread<0:
            raise ValueError('Invalid coarse TTA spread.')
        sides[ident]=min(maximum,base+factor*spread) if adaptive else base
    return centers,sides
