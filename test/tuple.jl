# Ensure elementwise comparisons against scalars behave the same as arrays
for fn in ((.==), (.!=), (.<=), (.>=), (.<), (.>))
    @test fn((1,2,3), 0) == fn([1,2,3], 0)
    @test fn((1,2,3), 1) == fn([1,2,3], 1)
    @test fn((1,2,3), 2) == fn([1,2,3], 2)
    @test fn((1,2,3), 3) == fn([1,2,3], 3)
    @test fn((1,2,3), 4) == fn([1,2,3], 4)
    
    @test fn((1,2,3), [0,1,2]) == fn([1,2,3], [0,1,2])
    @test fn((1,2,3), [1,2,3]) == fn([1,2,3], [1,2,3])
    @test fn((1,2,3), [2,3,4]) == fn([1,2,3], [2,3,4])
    
    @test fn((1,2,3), [2,1,0]) == fn([1,2,3], [2,1,0])
    @test fn((1,2,3), [3,2,1]) == fn([1,2,3], [3,2,1])
    @test fn((1,2,3), [4,3,2]) == fn([1,2,3], [4,3,2])
    @test fn((1,2,3), [5,4,3]) == fn([1,2,3], [5,4,3])
    @test fn((1,2,3), [6,5,4]) == fn([1,2,3], [6,5,4])
    
    @test_throws DimensionMismatch fn((1,2,3),(1,2))
    @test_throws DimensionMismatch fn((1,2,3),[1,2])
end
