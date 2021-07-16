! Manlin Chawla 01205586
! Homework 2

! This is a main program which can be used with the nmodel module
! if and as you wish as you develop your codes.
! It reads the full set of 70000 images and labels from data.csv
! into the variable, p, which is then split into the
! images (xfull) and labels (yfull)
! The subroutine sgd is called with the first d=100 images+labels;
! this portion of the code will only work after the snmodel subroutine
! in the nmodel module has been completed. Finally, the fitting parameters
! returned by sgd are written to the file, fvec.txt
! Note that here the intial fitting parameters are sampled from
! an uniform distribution while a normal distribution should be used
! in your python code.
! You should not submit this code with your assignment.
! To compile: gfortran -o main.exe hw2_dev.f90 hw2_main.f90

program hw2main
  use nmodel !brings in module variables x and y as well as the module subroutines
  implicit none
  integer, parameter :: n=784,dfull=70000, dtest=1000, m=0
  integer:: d=10000
  integer :: i1, p(n+1,dfull)
  real(kind=8) :: xfull(n,dfull),xtest(n,dtest)
  real(kind=8), allocatable, dimension(:) :: fvec0,fvec !array of fitting parameters
  integer :: yfull(dfull),ytest(dtest) !Labels
  real(kind=8) :: cost
  real(kind=8), dimension(n+1) :: cgrad !gradient of cost

  !read raw data from data.csv and store in p
  open(unit=12,file='data.csv')
  do i1=1,dfull
    read(12,*) p(:,i1)
    if (mod(i1,1000)==0) print *, 'read in image #', i1
  end do
  close(12)

  open(unit=13,file='p.dat')
  do i1=1,n
    write(13,*) p(i1,1)
  end do
  close(13)

  !Rearrange into input data, x,  and labels, y
  xfull = p(1:n,:)/255.d0
  yfull = p(n+1,:)
  yfull = mod(yfull,2)

  print *, 'yfull(1:4)',yfull(1:4) ! output first few labels
  xtest = xfull(:,dfull-dtest+1:dfull) ! set up test data (not used below)
  ytest = yfull(dfull-dtest+1:dfull)

  !SNM, d training images---------------

  if(m==0) then
    allocate(fvec(n+1), fvec0(n+1))
  else
    allocate(fvec0(m*(n+2)+1),fvec(m*(n+2)+1))
  end if
  call random_number(fvec0) !set initial fitting parameters
  fvec0 = fvec0-0.5d0
  call data_init(n, dtest)
  nm_x = xfull(:,dfull-dtest:)
  nm_y = yfull(dfull-dtest:)
  call snmodel(fvec0,n,dtest,cost,cgrad)
  print *, "Initial cost: ", cost

  call data_init(n,d) !allocate module variables x and y
  nm_x = xfull(:,1:d) !set module variables
  nm_y = yfull(1:d)
  !Use stochastic gradient descent, setting m>0 will use nnmodel instead of snmodel within sgd subroutine
  call sgd(fvec0,n,m,d,0.1d0,fvec) !requires snmodel subroutine to be operational


  call data_init(n, dtest)
  nm_x = xfull(:,dfull-dtest:)
  nm_y = yfull(dfull-dtest:)
  call snmodel(fvec,n,dtest,cost,cgrad)
  print *, "Final cost: ", cost
  !write fitting parameters to file, fvec.txt
  !Can be loaded into python with f = np.loadtxt('fvec.txt')
  open(unit=22,file='fvec.txt')
  do i1=1,n+1
    write(22,*) fvec(i1)
  end do
  close(22)
  deallocate(nm_x,nm_y)
end program hw2main
