use strict;
use warnings;

# TH10 0.02a Rank Select, crop (28,18,180,48). Sample the stable title strokes.
open my $file, '<:raw', $ARGV[0] or die "Could not read the rank-screen crop";
local $/;
my $data = <$file>;
$data =~ s/^P6\s+(?:\#[^\n]*\n\s*)*180\s+48\s+255\s// or die "Unexpected rank-screen pixel format";
die "Incomplete rank-screen pixels" unless length($data) == 180 * 48 * 3;
my @points = ([75,32], [140,30], [44,30], [147,40], [165,22], [105,10], [126,23],
              [111,39], [98,15], [69,22], [158,23], [133,22], [128,32], [117,39],
              [120,34], [64,32], [51,39]);
my $matches = 0;
for my $point (@points) {
    my ($x, $y) = @$point;
    my ($r, $g, $b) = unpack('C3', substr($data, 3 * (180 * $y + $x), 3));
    $matches++ if $r >= 220 && $g >= 220 && $b >= 220;
}
exit($matches >= 14 ? 0 : 1);
