import Base.Test: @test

# Cursory checks against comprehensions
test_cases = ((1, (i for i=1:10), [i for i=1:10]),
              (1, (i for i=-4:5), [i for i=-4:5]),
              (2, ("$i,$j" for i=1:10, j=1:10), ["$i,$j" for i=1:10, j=1:10]),
              (2, ("$i,$j" for i=-4:5, j=-4:5), ["$i,$j" for i=-4:5, j=-4:5]),
              (2, ("$i,$j" for i=-9:0, j=-4:5), ["$i,$j" for i=-9:0, j=-4:5]),
              (3, ("$i,$j,$k" for i=1:10, j=1:10, k=1:10), ["$i,$j,$k" for i=1:10, j=1:10, k=1:10]),
              (3, ("$i,$j,$k" for i=-4:5, j=-4:5, k=-4:5), ["$i,$j,$k" for i=-4:5, j=-4:5, k=-4:5]),
              (3, ("$i,$j,$k" for i=-9:0, j=-4:5, k=-3:3), ["$i,$j,$k" for i=-9:0, j=-4:5, k=-3:3]))

for (ndims, g, d) in test_cases
    @test full(g) == d
    @test size(g) == size(d)
    @test length(g) == length(d)
    
    @test g[2] == d[2]
    @test g[2,1] == d[2,1]
    if ndims > 1
        @test g[1,2] == d[1,2]
        @test g[2,3] == d[2,3]
        @test g[2,3,1] == d[2,3,1]
        @test g[12] == d[12]
    end
    if ndims > 2
        @test g[1,2,3] == d[1,2,3]
        @test g[2,3,2] == d[2,3,2]
        @test g[2,3,2,1] == d[2,3,2,1]
        @test g[102] == d[102]
    end
end
