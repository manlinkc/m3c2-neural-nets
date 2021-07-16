! Manlin Chawla 01205586
! M3C 2018 Homework 2

!This module contains two module variables and four subroutines;
!two of these routines must be developed for this assignment.
!Module variables--
! nm_x: training images, typically n x d with n=784 and d<=60000
! nm_y: labels for training images, d-element array containing 0s and 1s
! corresponding to images of even and odd integers, respectively.

!Module routines---
! data_init: allocate nm_x and nm_y using input variables n and d. Used by sgd, may be used elsewhere if needed
! sgd: Use simple stochastic descent algorithm to iteratively find fitting parameters using either snmodel (when m=0) or
! nnmodel (when m>1)
! snmodel: compute cost function and gradient using single neuron model (SNM) with nm_x and nm_y, and
! with fitting parameters provided as input
! nnmodel: compute cost function and gradient using neural network model (NNM) with 1 output neuron and
! m neurons in the inner layer, and with nm_x and nm_y, and with fitting parameters provided as input

module nmodel
  implicit none
  real(kind=8), allocatable, dimension(:,:) :: nm_x
  integer, allocatable, dimension(:) :: nm_y

contains

!---allocate nm_x and nm_y deallocating first if needed (used by sgd)---
subroutine data_init(n,d)
  implicit none
  integer, intent(in) :: n,d
  if (allocated(nm_x)) deallocate(nm_x)
  if (allocated(nm_y)) deallocate(nm_y)
  allocate(nm_x(n,d),nm_y(d))
end subroutine data_init


!Compute cost function and its gradient for single neuron model
!for d images (in nm_x) and d labels (in nm_y) along with the
!fitting parameters provided as input in fvec.
!The weight vector, w, corresponds to fvec(1:n) and
!the bias, b, is stored in fvec(n+1)
!Similarly, the elements of dc/dw should be stored in cgrad(1:n)
!and dc/db should be stored in cgrad(n+1)
!Note: nm_x and nm_y must be allocated and set before calling this subroutine.

!Single Neuron Model
subroutine snmodel(fvec,n,d,c,cgrad)
  implicit none
  integer, intent(in) :: n,d !training data sizes
  real(kind=8), dimension(n+1), intent(in) :: fvec !fitting parameters
  real(kind=8), intent(out) :: c !cost
  real(kind=8), dimension(n+1), intent(out) :: cgrad !gradient of cost

  !Declare other variables as needed
  real(kind=8), dimension(d) :: a, z, gamma, epsilon
  real(kind=8), dimension(n,d) :: dadw
  real(kind=8), dimension(n) :: dcdw
  real(kind=8) :: dcdb
  integer :: i1, i2, i3, i4, i5, i6, i7, i8, i9, i10

  !Compute z values
  z=matmul(fvec(1:n),nm_x) + fvec(n+1)

  !Calculate activations
  do i1=1,d
      a(i1)=1.d0/(1+exp(-z(i1)))
  end do

  !Compute cost function
  c=0.d0
  do i2=1,d
    c=c+(a(i2)-nm_y(i2))**2
  end do
  c=c/(2.d0*real(d))

  !Compute gamma values and epsilon
  do i3=1,d
    gamma(i3)=a(i3)*(1.d0-a(i3))
  end do
  epsilon=a-nm_y

  !Compute da/dw
  do i5=1,n
    do i6=1,d
      dadw(i5,i6)=nm_x(i5,i6)*gamma(i6)
    end do
  end do

  !Compute dc/db
  dcdb=dot_product(epsilon,gamma)/real(d)

  !Compute dcdw
  do i7=1,n
    dcdw(i7)=0.d0
    do i8=1,d
      dcdw(i7)=dcdw(i7)+epsilon(i8)*dadw(i7,i8)
    end do
  end do
  dcdw=dcdw/real(d)

  !Set cgrad
  do i9=1,n
    cgrad(i9)=dcdw(i9)
  end do
  cgrad(n+1)=dcdb

end subroutine snmodel

!!Compute cost function and its gradient for neural network model
!for d images (in nm_x) and d labels (in nm_y) along with the
!fitting parameters provided as input in fvec. The network consists of
!an inner layer with m neurons and an output layer with a single neuron.
!fvec contains the elements of dw_inner, b_inner, w_outer, and b_outer
!Code has been provided below to "unpack" fvec
!The elements of dc/dw_inner,dc/db_inner, dc/dw_outer,dc/db_outer should be stored in cgrad
!and should be "packed" in the same order that fvec was unpacked.
!Note: nm_x and nm_y must be allocated and set before calling this subroutine.

!Neural Network Model
subroutine nnmodel(fvec,n,m,d,c,cgrad)
implicit none
  integer, intent(in) :: n,m,d !training data and inner layer sizes
  real(kind=8), dimension(m*(n+2)+1), intent(in) :: fvec !fitting parameters
  real(kind=8), intent(out) :: c !cost
  real(kind=8), dimension(m*(n+2)+1), intent(out) :: cgrad !gradient of cost

  !Declare other variables as needed
  integer :: i1,j1,i2,i3,i4
  real(kind=8), dimension(m,n) :: w_inner
  real(kind=8), dimension(m) :: b_inner, w_outer, z_inner, a_inner, gamma_inner,daoutdwout, daoutdb
  real(kind=8) :: b_outer, z_outer, epsilon, gamma_outer, a_outer

  !Unpack fitting parameters (use if needed)
  do i1=1,n
    j1 = (i1-1)*m+1
    w_inner(:,i1) = fvec(j1:j1+m-1) !inner layer weight matrix
  end do

  b_inner = fvec(n*m+1:n*m+m)     !inner layer bias vector
  w_outer = fvec(n*m+m+1:n*m+2*m) !output layer weight vector
  b_outer = fvec(n*m+2*m+1)       !output layer bias

  !Initialize c and cgrad to use in summations
  c = 0.d0
  cgrad = 0.d0

  do i2 = 1,d

    do i1 = 1,m
      z_inner(i1) = dot_product(w_inner(i1,:),nm_x(:,i2)) + b_inner(i1) !inner layer z vector
      a_inner(i1) = 1.d0/(1.d0+exp(-z_inner(i1))) !inner layer activations
    end do

    gamma_inner = a_inner*(1.d0-a_inner) !inner layer gamma values

    z_outer = dot_product(w_outer,a_inner) + b_outer !outer layer z values
    a_outer = 1.d0/(1.d0+exp(-z_outer)) !outer layer activations

    c = c + (a_outer-nm_y(i2))**2  !summation to calculate the cost

    gamma_outer = a_outer*(1-a_outer) !outer layer gamma values, this is also daoutdbout

    !Derivatives
    !derivative of outer layer activation values by out layer weights
    !derivative of outer layer activation values by inner layer biases
    daoutdwout=gamma_outer*a_inner
    daoutdb=gamma_outer*gamma_inner*w_outer

    !Computing cgrad
    epsilon = a_outer - nm_y(i2)

    !Format of cgrad=[dcdwin, dcdbin, dcdwout, dcdbout]
    !Format of cgrad=[#elements=m*n, #elements=m , #elements=m, #elements=1]

    !dcdwin
    do i3=1,n
      i4 = (i3-1)*m+1
      cgrad(i4:i4+m-1) = cgrad(i4:i4+m-1) + epsilon*daoutdb*nm_x(i3,i2)
    end do

    !dcdbin
    cgrad(n*m+1:n*m+m) = cgrad(n*m+1:n*m+m) + epsilon*daoutdb

    !dcdwout
    cgrad(n*m+m+1:n*m+2*m) = cgrad(n*m+m+1:n*m+2*m) + epsilon*daoutdwout

    !dcdbout
    cgrad(m*(n+2)+1) = cgrad(m*(n+2)+1) + epsilon*gamma_outer !gamma_outer=daoutdbout
  end do

  !Final cost and cgrad
  c = c/(2.d0*real(d))
  cgrad = cgrad/(real(d))

end subroutine nnmodel

!Use crude implementation of stochastic gradient descent
!to move towards optimal fitting parameters using either
! snmodel or nnmodel. Iterates for 400 "epochs" and final fitting
!parameters are stored in fvec.
!Input:
!fvec_guess: initial vector of fitting parameters
!n: number of pixels in each image (should be 784)
!m: number of neurons in inner layer; snmodel is used if m=0
!d: number of training images to be used; only the 1st d images and labels stored
!in nm_x and nm_y are used in the optimization calculation
!alpha: learning rate, it is fine to keep this as alpha=0.1 for this assignment
!Output:
!fvec: fitting parameters, see comments above for snmodel and nnmodel to see how
!weights and biases are stored in the array.
!Note: nm_x and nm_y must be allocated and set before calling this subroutine.

subroutine sgd(fvec_guess,n,m,d,alpha,fvec)
  implicit none
  integer, intent(in) :: n,m,d
  real(kind=8), dimension(:), intent(in) :: fvec_guess
  real(kind=8), intent(in) :: alpha
  real(kind=8), dimension(size(fvec_guess)), intent(out) :: fvec
  integer :: i1, j1, i1max=400 !change back to 400
  real(kind=8) :: c
  real(kind=8), dimension(size(fvec_guess)) :: cgrad
  real(kind=8), allocatable, dimension(:,:) :: xfull
  integer, allocatable, dimension(:) :: yfull
  real(kind=8), dimension(d) :: a
  real(kind=8), dimension(d+1) :: r
  integer, dimension(d+1) :: j1array

  !store full nm_x,nm_y
  print *, size(nm_x,1),size(nm_x,2)
  allocate(xfull(size(nm_x,1),size(nm_x,2)),yfull(size(nm_y)))
  xfull = nm_x
  yfull = nm_y
  !will only use one image at a time, so need to reallocate nm_x,nm_y
  call data_init(n,1)
  fvec = fvec_guess

  do i1=1,i1max
    call random_number(r)
    j1array = floor(r*d+1.d0) !d random integers falling between 1 and d (inclusive); will compute c, cgrad for one image at a time cycling through these integers

    do j1 = 1,d
      nm_x(:,1) = xfull(:,j1array(j1))
      nm_y = yfull(j1array(j1))

      !compute cost and gradient with randomly selected image
      if (m==0) then
        call snmodel(fvec,n,1,c,cgrad)
      else
        call nnmodel(fvec,n,m,1,c,cgrad)
      end if
      fvec = fvec - alpha*cgrad !update fitting parameters using gradient descent step
    end do

    if (mod(i1,50)==0) print *, 'completed epoch # ', i1, c

  end do

 !reset nm_x,nm_y to intial state at beginning of subroutine
  call data_init(size(xfull,1),size(xfull,2))
  nm_x = xfull
  nm_y = yfull
  deallocate(xfull,yfull)

end subroutine sgd

end module nmodel
