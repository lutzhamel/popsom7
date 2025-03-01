!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! an implementation of the stochastic SOM training algorithm based on
! ideas from tensor algebra
! written by Lutz Hamel, University of Rhode Island (c) 2016
!
! LICENSE: This program is free software; you can redistribute it and/or modify it
! under the terms of the GNU General Public License as published by the Free Software
! Foundation.
!
! This program is distributed in the hope that it will be useful, but WITHOUT ANY
! WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
! PARTICULAR PURPOSE. See the GNU General Public License for more details.
!
! A copy of the GNU General Public License is available at
! http://www.r-project.org/Licenses/
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!! vsom !!!!!!
subroutine vsom(neurons,dt,dtrows,dtcols,xdim,ydim,alpha,train,dtix)
    use, intrinsic :: ISO_FORTRAN_ENV, only: real32, int32
    implicit none

    !!! Input/Output
    ! NOTE: neurons are assumed to be initialized to small random values and then trained.
    integer(kind=int32),intent(in)  :: dtrows,dtcols,xdim,ydim,train
    real(kind=real32),intent(in)    :: alpha
    real(kind=real32),intent(inout) :: neurons(1:xdim*ydim,1:dtcols)
    real(kind=real32),intent(in)    :: dt(1:dtrows,1:dtcols)
    integer(kind=int32),intent(in)  :: dtix(1:train)

    !!! Locals
    ! Note: the neighborhood cache is only valid as long as cache_counter < nsize_step
    integer(kind=int32) :: step_counter
    integer(kind=int32) :: nsize
    integer(kind=int32) :: nsize_step
    integer(kind=int32) :: epoch
    integer(kind=int32) :: i
    integer(kind=int32) :: ca(1)
    integer(kind=int32) :: c
    real(kind=real32)   :: cache(1:xdim*ydim,1:xdim*ydim)       ! neighborhood cache
    logical             :: cache_valid(1:xdim*ydim)
    real(kind=real32)   :: diff(1:xdim*ydim,1:dtcols)
    real(kind=real32)   :: squ(1:xdim*ydim,1:dtcols)
    real(kind=real32)   :: s(1:xdim*ydim)
    integer(kind=int32) :: coord_lookup(1:xdim*ydim,1:2)
    integer(kind=int32) :: ix
    real(kind=real32)   :: beta

    !!! setup
    nsize = max(xdim,ydim) + 1
    nsize_step = ceiling((train*1.0)/nsize)
    step_counter = 0
    cache_valid = .false.

    ! fill the 2D coordinate lookup table that associates each
    ! 1D neuron coordinate with a 2D map coordinate
    do i=1,xdim*ydim
        call coord2D(coord_lookup(i,:),i,xdim)
    end do
    
    !!! training !!!
    ! the epochs loop
    do epoch=1,train

        step_counter = step_counter + 1
        if (step_counter == nsize_step) then
            step_counter = 0
            nsize = nsize - 1
            cache_valid = .false.
        endif

        ! select a training observation
        ix = dtix(epoch)

        !!! competetive step
        diff = neurons - spread(dt(ix, :), 1, size(neurons, 1))
        squ = diff * diff
        s = sum(squ, dim=2)
        ca = minloc(s)
        c = ca(1)

        !!! update step
        ! compute neighborhood vector
        call Gamma(cache(:,c),cache_valid,coord_lookup,nsize,xdim,ydim,c)
        beta = 1.0 - real(epoch, kind=real32)/real(train, kind=real32)
        neurons = neurons - spread(cache(:, c) * alpha * beta, 2, dtcols) * diff
    enddo
    return
end subroutine vsom


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!! Gamma !!!!!!
subroutine Gamma(neighborhood,cache_valid,coord_lookup,nsize,xdim,ydim,c)
    use, intrinsic :: ISO_FORTRAN_ENV, only: real32, int32
    implicit none

    ! parameters
    integer(kind=int32),intent(in)   :: nsize,xdim,ydim,c
    real(kind=real32),intent(inout)  :: neighborhood(1:xdim*ydim)
    logical,intent(inout)            :: cache_valid(1:xdim*ydim)
    integer(kind=int32),intent(in)   :: coord_lookup(1:xdim*ydim,1:2)

    ! locals
    integer(kind=int32) :: c2D(1:2)
    real(kind=real32)   :: d(1:xdim*ydim)

    ! cache is valid - nothing to do
    if (cache_valid(c)) then
        return
    endif

    ! convert the 1D neuron index into a 2D map index
    call coord2D(c2D,c,xdim)

    ! for each neuron m check if on the grid it is
    ! within the neighborhood.
    ! Compute squared distances from c2D to each row of coord_lookup
    d = sum((coord_lookup - spread(c2D, 1, size(coord_lookup, 1)))**2, dim=2)
    ! Set neighborhood to 1.0 where the distance is below the threshold, 0.0 otherwise
    neighborhood = merge(1.0_real32, 0.0_real32, d < (nsize*1.5)**2)
    
    ! cache it
    cache_valid(c) = .true.

    return
end subroutine Gamma

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!! coord2D - convert a 1D rowindex into a 2D map coordinate !!!
pure subroutine coord2D(coord,ix,xdim)
    use, intrinsic :: ISO_FORTRAN_ENV, only: int32
    implicit none

    integer(kind=int32),intent(out) :: coord(1:2)
    integer(kind=int32),intent(in)  :: ix,xdim

    coord(1) = modulo(ix-1,xdim) + 1
    coord(2) = (ix-1)/xdim + 1

    return
end subroutine coord2D

