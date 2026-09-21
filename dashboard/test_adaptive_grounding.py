"""Small regressions for entity binding; no ML/provider calls."""
from unittest.mock import patch
from django.test import SimpleTestCase
from spatial.schema import Location
from .services.grounding import prepare,patterns,aliases,Clarification


class GroundingTests(SimpleTestCase):
    def setUp(self):
        aliases.cache_clear();patterns.cache_clear()
        self.addCleanup(aliases.cache_clear);self.addCleanup(patterns.cache_clear)
        self.rows=[dict(id='osm:node:1',name='مستشفى التجربة',aliases=['مستشفى التجربة'],lat=24.7,lon=46.6,category='مستشفى')]
        self.points=(Location('osm:node:1','مستشفى التجربة',24.7,46.6,'مستشفى'),)
        self.index=patch('dashboard.services.grounding.index',return_value=self.rows);self.index.start();self.addCleanup(self.index.stop)
        self.locations=patch('dashboard.services.grounding.locations',return_value=self.points);self.locations.start();self.addCleanup(self.locations.stop)

    def test_explicit_reference_wins_over_direction_target(self):
        text,context=prepare('ما اتجاه مستشفى التجربة بالنسبة إلى وسط الرياض؟')
        self.assertEqual(context.anchor.name,'وسط الرياض')
        self.assertIn('«مستشفى التجربة»',text)

    def test_unknown_quoted_reference_does_not_use_city_default(self):
        with self.assertRaises(Clarification):prepare('ما أقرب صيدلية إلى «اسم غير موجود»؟')

    def test_named_reference_overrides_optional_coordinates(self):
        _,context=prepare('ما أقرب صيدلية إلى مستشفى التجربة؟',{'lat':25,'lon':47})
        self.assertEqual(context.anchor.identity,'osm:node:1')
