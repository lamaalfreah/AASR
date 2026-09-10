"""Audited spherical geometry. Coordinates in degrees, distances in km."""
import math

def valid_coord(lat, lon):
    return math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180


def distance(a, b, radius=6371.0088):
    p, q = math.radians(a[0]), math.radians(b[0])
    dp, dl = q-p, math.radians(b[1]-a[1])
    h = math.sin(dp/2)**2 + math.cos(p)*math.cos(q)*math.sin(dl/2)**2
    return 2*radius*math.asin(math.sqrt(max(0.0, min(1.0, h))))


def bearing(a, b):
    p, q = math.radians(a[0]), math.radians(b[0]); dl = math.radians(b[1]-a[1])
    return math.degrees(math.atan2(math.sin(dl)*math.cos(q),
           math.cos(p)*math.sin(q)-math.sin(p)*math.cos(q)*math.cos(dl))) % 360


def direction(b, sectors=4):
    labels = ['شمال', 'شرق', 'جنوب', 'غرب'] if sectors == 4 else [
        'شمال', 'شمال شرق', 'شرق', 'جنوب شرق', 'جنوب', 'جنوب غرب', 'غرب', 'شمال غرب']
    return labels[int((b+180/sectors)//(360/sectors)) % sectors]

