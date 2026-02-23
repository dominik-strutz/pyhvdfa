! ==============================================================================
! hv_wrapper.f90 — Python/f2py entry point for HV-DFA
!
! This file is the single compilation unit for the pyhvdfa Python extension.
! It INCLUDEs all HV-DFA source files (unchanged) and adds two new procedures:
!   SET_MODEL_PARAMETERS  — verbatim from HV.f90 (lives outside PROGRAM there)
!   COMPUTE_HV            — the public API replacing PROGRAM HV
!
! Original HV-DFA code:
!   (c) Antonio Garcia Jerez, Jose Pina Flores (UNAM / University of Almeria)
!   https://doi.org/10.1093/gji/ggu005
! ==============================================================================

      ! ---- HV-DFA source files (unchanged) ----
      INCLUDE './HV-DFA/modules.f90'
      INCLUDE './HV-DFA/aux_procedures.f90'
      INCLUDE './HV-DFA/Dispersion.f90'
      INCLUDE './HV-DFA/RootSolverTemplate.f90'
      INCLUDE './HV-DFA/y.f90'
      INCLUDE './HV-DFA/GR_new.f90'
      INCLUDE './HV-DFA/GL.f90'
      INCLUDE './HV-DFA/WangMethod_Rayleigh_new.f90'
      INCLUDE './HV-DFA/WangMethod_Love.f90'
      INCLUDE './HV-DFA/BW_INTEGRALS_DAMPED.f90'

! ==============================================================================
!  SET_MODEL_PARAMETERS
!  Verbatim copy from the end of HV.f90 (originally defined after PROGRAM HV).
! ==============================================================================
      SUBROUTINE SET_MODEL_PARAMETERS()
      USE Marc,ONLY:NCAPAS,G_MAXRAYLEIGHSLOWNESS,G_SLOWS,G_SLOWP,ALFA,BTA, &
                    G_VELOCITYINVERSION,G_SLOWSMIN,G_SLOWSMAX,HALFSPACE_RAYLEIGH
      IMPLICIT NONE
      INTEGER I,IMAXSLOWS
      G_SLOWS=1/REAL(BTA)
      G_SLOWP=1/REAL(ALFA)
      IMAXSLOWS=MAXLOC(G_SLOWS,1)
      G_MAXRAYLEIGHSLOWNESS=REAL(HALFSPACE_RAYLEIGH(IMAXSLOWS))
      G_SLOWSMAX=G_SLOWS(IMAXSLOWS)
      G_SLOWSMIN=G_SLOWS(NCAPAS) ! MINIMUM SLOWNESS FOR THE HIGHER MODES
      DO I=2,NCAPAS
        IF(G_SLOWS(I)>G_SLOWS(I-1).OR.G_SLOWP(I)>G_SLOWP(I-1))THEN
          G_VELOCITYINVERSION=I;RETURN;
        ENDIF
      ENDDO
      G_VELOCITYINVERSION=-1;RETURN;
      END SUBROUTINE SET_MODEL_PARAMETERS

! ==============================================================================
!  COMPUTE_HV — main Python entry point
!
!  Replaces PROGRAM HV: accepts model + control parameters directly instead of
!  reading them from the command line, and returns the H/V array instead of
!  writing it to a file.
!
!  Arguments (all intent(in) except hv_out):
!    ncapas_in  — number of layers (including half-space)
!    alfa_in    — P-wave velocities [m/s], shape (ncapas_in,), float64
!    bta_in     — S-wave velocities [m/s], shape (ncapas_in,), float64
!    h_in       — layer thicknesses [m],  shape (ncapas_in,), float64
!                 (last element is ignored; half-space has no thickness)
!    rho_in     — layer densities [kg/m³], shape (ncapas_in,), float64
!    nx_in      — number of frequency samples (hidden from Python, derived from x_in)
!    x_in       — angular frequencies ω = 2π·f [rad/s], shape (nx_in,), float32
!    nmr        — maximum number of Rayleigh modes to search (≥1 recommended)
!    nml        — maximum number of Love modes to search  (0 = disable Love)
!    nks_in     — number of wavenumber integration points for body-wave part
!                 (0 = skip body-wave integrals; ≥50 recommended when used)
!    shdamp_in  — imaginary frequency fraction for SH body-wave integration
!                 (typical: 1e-5)
!    psvdamp_in — imaginary frequency fraction for PSV body-wave integration
!                 (typical: 1e-5)
!    prec_in    — slowness root-search precision [%]  (typical: 1e-4)
!  Output:
!    hv_out     — H/V spectral ratio at each frequency, shape (nx_in,), float64
! ==============================================================================
      SUBROUTINE COMPUTE_HV(NCAPAS_IN,ALFA_IN,BTA_IN,H_IN,RHO_IN,   &
                             NX_IN,X_IN,NMR,NML,NKS_IN,              &
                             SHDAMP_IN,PSVDAMP_IN,PREC_IN,            &
                             HV_OUT)
      USE TYPES
      USE Marc,ONLY:VALUES_L,VALUES_R,VALID_L,VALID_R,G_NX,X,        &
                    ALFA,BTA,H,RHO,MU,NCAPAS,G_NMODES,ISRAYLEIGH,    &
                    G_DX,G_PRECISION,G_SLOWS,G_SLOWP
      USE Globales,ONLY:SHDAMP,PSVDAMP,PI
      IMPLICIT NONE

      ! ---- Input arguments ----
      INTEGER,          INTENT(IN) :: NCAPAS_IN
      REAL(LONG_FLOAT), INTENT(IN) :: ALFA_IN(NCAPAS_IN)
      REAL(LONG_FLOAT), INTENT(IN) :: BTA_IN(NCAPAS_IN)
      REAL(LONG_FLOAT), INTENT(IN) :: H_IN(NCAPAS_IN)
      REAL(LONG_FLOAT), INTENT(IN) :: RHO_IN(NCAPAS_IN)
      INTEGER,          INTENT(IN) :: NX_IN
      REAL,             INTENT(IN) :: X_IN(NX_IN)
      INTEGER,          INTENT(IN) :: NMR
      INTEGER,          INTENT(IN) :: NML
      INTEGER,          INTENT(IN) :: NKS_IN
      REAL,             INTENT(IN) :: SHDAMP_IN
      REAL,             INTENT(IN) :: PSVDAMP_IN
      REAL,             INTENT(IN) :: PREC_IN

      ! ---- Output argument ----
      REAL(LONG_FLOAT), INTENT(OUT) :: HV_OUT(NX_IN)

      ! ---- Local variables ----
      INTEGER INDX,INDXX
      INTEGER NM_R,NM_L,NM_RL
      LOGICAL,POINTER   :: SLOWNESSVALID(:)
      REAL,POINTER      :: SLOWNESSVALUES(:)
      LOGICAL SALIDA
      REAL(LONG_FLOAT),ALLOCATABLE :: G1(:,:),IMG2(:,:),IMG3(:,:)
      REAL(LONG_FLOAT),ALLOCATABLE :: IMG11_pihalf(:),IMG33(:)
      REAL(LONG_FLOAT),ALLOCATABLE :: IMVV(:),IMHPSV(:),IMHSH(:)
      INTEGER,ALLOCATABLE :: OFFSET_R(:),OFFSET_L(:)

      INTERFACE DISPERSION
        LOGICAL FUNCTION DISPERSION(VALUES,VALID) RESULT(RETORNO)
        REAL,INTENT(INOUT),DIMENSION(:),TARGET::VALUES
        LOGICAL,INTENT(INOUT),DIMENSION(:),TARGET::VALID
        END FUNCTION
      END INTERFACE

      ! ---- Initialise counters ----
      NM_R=0;NM_L=0;NM_RL=0

      ! ---- Populate module Marc -------------------------------------------
      NCAPAS=NCAPAS_IN
      G_NX=NX_IN

      ! Free any previous allocations left from earlier calls
      IF(ALLOCATED(ALFA))   DEALLOCATE(ALFA)
      IF(ALLOCATED(BTA))    DEALLOCATE(BTA)
      IF(ALLOCATED(H))      DEALLOCATE(H)
      IF(ALLOCATED(RHO))    DEALLOCATE(RHO)
      IF(ALLOCATED(MU))     DEALLOCATE(MU)
      IF(ALLOCATED(X))      DEALLOCATE(X)
      IF(ALLOCATED(G_SLOWS))DEALLOCATE(G_SLOWS)
      IF(ALLOCATED(G_SLOWP))DEALLOCATE(G_SLOWP)
      IF(ALLOCATED(VALUES_R))DEALLOCATE(VALUES_R)
      IF(ALLOCATED(VALID_R)) DEALLOCATE(VALID_R)
      IF(ALLOCATED(VALUES_L))DEALLOCATE(VALUES_L)
      IF(ALLOCATED(VALID_L)) DEALLOCATE(VALID_L)

      ALLOCATE(ALFA(NCAPAS_IN),BTA(NCAPAS_IN),H(NCAPAS_IN),          &
               RHO(NCAPAS_IN),MU(NCAPAS_IN))
      ALLOCATE(X(NX_IN))
      ALLOCATE(G_SLOWS(NCAPAS_IN),G_SLOWP(NCAPAS_IN))

      ALFA = ALFA_IN
      BTA  = BTA_IN
      H    = H_IN
      RHO  = RHO_IN
      MU   = RHO_IN * BTA_IN * BTA_IN    ! shear modulus μ = ρ·Vs²

      X    = X_IN                         ! ω values (single precision in Marc)

      ! ---- Populate module Globales ----------------------------------------
      SHDAMP  = SHDAMP_IN
      PSVDAMP = PSVDAMP_IN

      ! ---- Derive slowness limits from model --------------------------------
      CALL SET_MODEL_PARAMETERS

      ! ==== RAYLEIGH DISPERSION CURVES ======================================
      G_NMODES  = NMR
      G_PRECISION = PREC_IN * 1.E-2
      G_DX      = 0.1

      ALLOCATE(VALUES_R(G_NX*G_NMODES),VALID_R(G_NX*G_NMODES))
      VALUES_R=0;VALID_R=.FALSE.
      ISRAYLEIGH=.TRUE.
      SALIDA=DISPERSION(VALUES_R,VALID_R)

      ! Determine the cutoff frequencies and actual number of Rayleigh modes
      ALLOCATE(OFFSET_R(G_NMODES))
      OFFSET_R=-1
      NM_R=G_NMODES        ! default: all tried modes found
      INDX=1
      DO INDXX=1,G_NMODES
        SLOWNESSVALUES=>VALUES_R((INDXX-1)*G_NX+1:INDXX*G_NX)
        SLOWNESSVALID =>VALID_R ((INDXX-1)*G_NX+1:INDXX*G_NX)
        DO WHILE((.NOT.SLOWNESSVALID(INDX)).AND.(INDX<G_NX))
          INDX=INDX+1
        ENDDO
        IF(SLOWNESSVALID(INDX))THEN
          OFFSET_R(INDXX)=INDX-1
        ELSE
          NM_R=INDXX-1
          EXIT
        ENDIF
      ENDDO

      ! ==== LOVE DISPERSION CURVES ==========================================
      IF(NML>0)THEN
        G_NMODES  = NML
        G_PRECISION = PREC_IN * 1.E-2
        G_DX      = 0.1

        ALLOCATE(VALUES_L(G_NX*G_NMODES),VALID_L(G_NX*G_NMODES))
        VALUES_L=0;VALID_L=.FALSE.
        ISRAYLEIGH=.FALSE.
        SALIDA=DISPERSION(VALUES_L,VALID_L) .AND. SALIDA

        ALLOCATE(OFFSET_L(G_NMODES))
        OFFSET_L=-1
        NM_L=G_NMODES
        ! Reset INDX before Love scan so we start from the lowest frequency
        INDX=1
        DO INDXX=1,G_NMODES
          SLOWNESSVALUES=>VALUES_L((INDXX-1)*G_NX+1:INDXX*G_NX)
          SLOWNESSVALID =>VALID_L ((INDXX-1)*G_NX+1:INDXX*G_NX)
          DO WHILE((.NOT.SLOWNESSVALID(INDX)).AND.(INDX<G_NX))
            INDX=INDX+1
          ENDDO
          IF(SLOWNESSVALID(INDX))THEN
            OFFSET_L(INDXX)=INDX-1
          ELSE
            NM_L=INDXX-1
            EXIT
          ENDIF
        ENDDO
      ENDIF

      ! ==== SW PART OF GREEN'S FUNCTIONS ====================================
      NM_RL=MAX(NM_R,NM_L)

      ! Allocate at least size-1 in the mode dimension to avoid zero-size arrays
      ALLOCATE(G1  (G_NX,MAX(NM_RL,1)))
      ALLOCATE(IMG2(G_NX,MAX(NM_RL,1)))
      ALLOCATE(IMG3(G_NX,MAX(NM_RL,1)))
      ALLOCATE(IMG11_pihalf(G_NX),IMG33(G_NX))
      G1=0.;IMG2=0.;IMG3=0.
      IMG11_pihalf=0.D0;IMG33=0.D0

      ! Rayleigh modal contributions
      DO INDXX=1,NM_R
        CALL GR(G1(:,INDXX),IMG3(:,INDXX),INDXX,OFFSET_R(INDXX))
        IMG11_pihalf(OFFSET_R(INDXX)+1:G_NX) =                         &
             IMG11_pihalf(OFFSET_R(INDXX)+1:G_NX)                       &
             + 0.5D0*REAL(G1(OFFSET_R(INDXX)+1:G_NX,INDXX),LONG_FLOAT)
        IMG33(OFFSET_R(INDXX)+1:G_NX) =                                 &
             IMG33(OFFSET_R(INDXX)+1:G_NX)                               &
             + IMG3(OFFSET_R(INDXX)+1:G_NX,INDXX)
      ENDDO

      ! Love modal contributions
      IF(NML>0)THEN
        DO INDXX=1,NM_L
          CALL GL(IMG2(:,INDXX),INDXX,OFFSET_L(INDXX))
          IMG11_pihalf(OFFSET_L(INDXX)+1:G_NX) =                       &
               IMG11_pihalf(OFFSET_L(INDXX)+1:G_NX)                     &
               - 0.5D0*IMG2(OFFSET_L(INDXX)+1:G_NX,INDXX)
        ENDDO
      ENDIF

      ! ==== BODY-WAVE INTEGRALS =============================================
      ALLOCATE(IMVV(G_NX),IMHPSV(G_NX),IMHSH(G_NX))
      IF(NKS_IN>0)THEN
        !$OMP PARALLEL DO
        DO INDXX=1,G_NX
          CALL BWR(IMVV(INDXX),IMHPSV(INDXX),IMHSH(INDXX),NKS_IN,X(INDXX))
        ENDDO
        !$OMP END PARALLEL DO
      ELSE
        IMVV=0.D0;IMHPSV=0.D0;IMHSH=0.D0
      ENDIF

      ! ==== ASSEMBLE H/V RATIO =============================================
      DO INDXX=1,G_NX
        HV_OUT(INDXX)=SQRT(2.D0*(IMG11_pihalf(INDXX)+IMHPSV(INDXX)+IMHSH(INDXX)) &
                           /(IMG33(INDXX)+IMVV(INDXX)))
      ENDDO

      ! ==== CLEANUP =========================================================
      DEALLOCATE(G1,IMG2,IMG3,IMG11_pihalf,IMG33)
      DEALLOCATE(IMVV,IMHPSV,IMHSH)
      DEALLOCATE(VALUES_R,VALID_R,OFFSET_R)
      IF(ALLOCATED(VALUES_L))DEALLOCATE(VALUES_L)
      IF(ALLOCATED(VALID_L)) DEALLOCATE(VALID_L)
      IF(ALLOCATED(OFFSET_L))DEALLOCATE(OFFSET_L)
      DEALLOCATE(ALFA,BTA,H,RHO,MU,X,G_SLOWS,G_SLOWP)

      END SUBROUTINE COMPUTE_HV
