
C***************************************************************************
C
C   IMSL ROUTINE NAME   - MERRCZ                                        
C                                                                       
C-----------------------------------------------------------------------
C                                                                       
C   COMPUTER            - IBM/DOUBLE                                    
C                                                                       
C   LATEST REVISION     - JUNE 1, 1982                                  
C                                                                       
C   PURPOSE             - EVALUATE A FUNCTION RELATED TO THE            
C                           COMPLEMENTED ERROR FUNCTION FOR A           
C                           COMPLEX ARGUMENT                            
C                                                                       
C   USAGE               - CALL MERRCZ (Z,W,IER)                         
C                                                                       
C   ARGUMENTS    Z      - INPUT COMPLEX ARGUMENT.  LET X AND Y          
C                           REPRESENT THE REAL AND IMAGINARY PARTS OF Z 
C                           RESPECTIVELY.  THEN X**2 + Y**2 MUST BE     
C                           LESS THAN MACHINE INFINITY.  ALSO, IF Z IS  
C                           LOCATED IN THE SECOND, THIRD OR FOURTH      
C                           QUADRANT OF THE COMPLEX PLANE, OR ON THE    
C                           BOUNDARY OF THE QUADRANTS, THEN Y**2 - X**2 
C                           MUST BE LESS THAN OR EQUAL TO THE LARGEST   
C                           ACCEPTABLE ARGUMENT FOR THE FORTRAN         
C                           EXPONENTIAL FUNCTION.                       
C
C                W      - OUTPUT COMPLEX VALUE: exp(-z**2) * erfc(-i*z)
C
C                IER    - ERROR PARAMETER. (OUTPUT)                     
C                         TERMINAL ERROR                                
C                           IER = 129 INDICATES THAT THE ABSOLUTE VALUE 
C                             OF INPUT ARGUMENT Z IS GREATER THAN THE   
C                             SQUARE ROOT OF MACHINE INFINITY.  W IS SET
C                             TO 0.                                     
C                           IER = 130 INDICATES THAT Z = (X,Y) IS NOT IN
C                             THE FIRST QUADRANT AND Y**2 - X**2 IS     
C                             GREATER THAN THE LARGEST ACCEPTABLE       
C                             ARGUMENT FOR THE FORTRAN EXPONENTIAL      
C                             FUNCTION. W IS SET TO MACHINE INFINITY.   
C                                                                       
C   PRECISION/HARDWARE  - DOUBLE/H32,H36                                
C                       - SINGLE/H48,H60                                
C                                                                       
C   REQD. IMSL ROUTINES - UERTST,UGETIO                                 
C                                                                       
C   NOTATION            - INFORMATION ON SPECIAL NOTATION AND           
C                           CONVENTIONS IS AVAILABLE IN THE MANUAL      
C                           INTRODUCTION OR THROUGH IMSL ROUTINE UHELP  
C                                                                       
C   REMARKS       ON THOSE MACHINES WHICH DO NOT OFFER THE APPROPRIATE  
C                 COMPLEX DATA TYPE, INPUT ARGUMENT Z AND OUTPUT        
C                 ARGUMENT W ARE TREATED AS VECTORS OF LENGTH 2.  THE   
C                 USER SHOULD DIMENSION THEM AS SUCH ON THOSE MACHINES. 
C                                                                       
C   COPYRIGHT           - 1982 BY IMSL, INC. ALL RIGHTS RESERVED.       
C                                                                       
C   WARRANTY            - IMSL WARRANTS ONLY THAT IMSL TESTING HAS BEEN 
C                           APPLIED TO THIS CODE. NO OTHER WARRANTY,    
C                           EXPRESSED OR IMPLIED, IS APPLICABLE.        
C                                                                       
C-----------------------------------------------------------------------
C                                                                       
      SUBROUTINE MERRCZ (Z,W,IER)                                       
C                                  SPECIFICATIONS FOR ARGUMENTS         
      INTEGER            IER                                            
      COMPLEX*16         Z,W                                            
C                                  SPECIFICATIONS FOR LOCAL VARIABLES   
      INTEGER            IFLAG,N,NCAPN,NM1,NN,NP1,NU,NUP1,NUP2          
      DOUBLE PRECISION   C,C1,H,H2,R1,R2,RE,RIMAG,RLMBDA,S,S1,S2,SRINF, 
     *                   T1,T2,X,XINF,XLARGE,XSMALL,XX,Y,YY,XSMARG,     
     *                   SRSMAG                                         
      DOUBLE PRECISION   DIMAG,DREAL                                    
      COMPLEX*16         ZDUM                                           
      COMPLEX*16         Q,WNU,WW,WZ,ZNU,ZSQ                            
      LOGICAL            B                                              
      DATA               XSMALL/ -710.d0 /
      DATA               XLARGE/ 709.d0 /
      DATA               XINF/ 1.d+100 /
      DATA               SRINF/ 1.d+50 /
      DATA               XSMARG/ 1.d-100 /
      DREAL(ZDUM) = ZDUM                                                
      DIMAG(ZDUM) = (0.D0,-1.D0)*ZDUM                                   
C                                  FIRST EXECUTABLE STATEMENT           
C-----------------------------------------------------------------------
CALL LIB MONITOR FROM MERRCZ, MAINTENANCE NUMBER 2899, DATE 82328       
C       CALL LIBMON( 2899 )                                             
C***PLEASE DON'T REMOVE OR CHANGE THE ABOVE CALL.  IT IS YOUR ONLY      
C***PROTECTION AGAINST YOUR USING AN OUT-OF-DATE OR INCORRECT           
C***VERSION OF THE ROUTINE.  THE LIBRARY MONITOR REMOVES THIS CALL,     
C***SO IT ONLY OCCURS ONCE, ON THE FIRST ENTRY TO THIS ROUTINE.         
C-----------------------------------------------------------------------

      ier = 0
      C1 = 1.12837916709551D0                                           
      B = .FALSE.                                                       
      X = DREAL(Z)                                                      
      Y = DIMAG(Z)                                                      
      IF(DABS(X).GT.SRINF .OR. DABS(Y).GT.SRINF) GO TO 5                
      SRSMAG = DSQRT(XSMARG) * 5.5D0                                    
      IF(DABS(X).LT.SRSMAG) X = 0.D0                                    
      IF(DABS(Y).LT.SRSMAG) Y = 0.D0                                    
      XX = X*X                                                          
      YY = XINF-(Y*Y)                                                   
      IF(XX .LE. YY) GO TO 10                                           
    5 IER = 129                                                         
      W = DCMPLX(0.D0,0.D0)                                             
      GO TO 9000                                                        
   10 XX = Y*Y-X*X                                                      
      IF(XX.LE.XLARGE) GO TO 15                                         
      IER = 130                                                         
      W = DCMPLX(XINF,XINF)                                             
      GO TO 9000                                                        
   15 XX = X                                                            
      YY = Y                                                            
      IFLAG = 1                                                         
      IF(X.GE.0.D0) GO TO 20                                            
      XX = -X                                                           
      IFLAG = 2                                                         
C                                  X IS NEGATIVE                        
   20 IF(Y.GE.0.D0) GO TO 25                                            
      YY = -Y                                                           
      IF(IFLAG.EQ.1) IFLAG = 3                                          
      IFLAG = IFLAG+1                                                   
   25 IF(YY.GE.4.29D0 .OR. XX.GE.5.33D0) GO TO 30                       
      S = (1.D0-YY/4.29D0)*DSQRT(1.D0-XX*XX/28.41D0)                    
      H = 1.6D0*S                                                       
      H2 = H+H                                                          
      NCAPN = 6.D0+23.D0*S                                              
      RLMBDA = H2**NCAPN                                                
      NU = 9.D0+21.D0*S                                                 
      GO TO 35                                                          
   30 H = 0.D0                                                          
      NCAPN = 0                                                         
      NU = 8                                                            
   35 IF(H.EQ.0.D0.OR.RLMBDA.EQ.0.D0) B = .TRUE.                        
      R1 = 0.D0                                                         
      R2 = 0.D0                                                         
      S1 = 0.D0                                                         
      S2 = 0.D0                                                         
      NUP1 = NU+1                                                       
      NUP2 = NU+2                                                       
      DO 40  NN=1,NUP1                                                  
         N = NUP2-NN                                                    
         NM1 = N-1                                                      
         T1 = YY+H+N*R1                                                 
         T2 = XX-N*R2                                                   
         C = .5D0/(T1*T1+T2*T2)                                         
         R1 = C*T1                                                      
         R2 = C*T2                                                      
         IF(H.LE.0.D0.OR.NM1.GT.NCAPN) GO TO 40                         
         T1 = RLMBDA+S1                                                 
         S1 = R1*T1-R2*S2                                               
         S2 = R2*T1+R1*S2                                               
         RLMBDA = RLMBDA/H2                                             
   40 CONTINUE                                                          
      IF(B) GO TO 45                                                    
      RE = C1*S1                                                        
      RIMAG = C1*S2                                                     
      GO TO 50                                                          
   45 RE = C1*R1                                                        
      RIMAG = C1*R2                                                     
   50 IF(YY.EQ.0.D0) RE = 0.D0                                          
      IF(YY.EQ.0.D0 .AND. -XX*XX.GT.XSMALL) RE = DEXP(-XX*XX)           
      IF(IFLAG.EQ.1) GO TO 60                                           
      ZNU = DCMPLX(XX,YY)                                               
      WZ = DCMPLX(RE,RIMAG)                                             
      ZSQ = ZNU*ZNU                                                     
      WNU = CDEXP(-ZSQ)                                                 
      WW = WNU+WNU                                                      
      Q = WW-WZ                                                         
      IF(IFLAG.EQ.3) GO TO 55                                           
      RE = DREAL(Q)                                                     
      RIMAG = -DIMAG(Q)                                                 
      IF(IFLAG.EQ.4) GO TO 60                                           
      ZNU = DCMPLX(XX,-YY)                                              
      ZSQ = ZNU*ZNU                                                     
      WNU = CDEXP(-ZSQ)                                                 
      WW = WNU+WNU                                                      
      Q = WW-DCMPLX(RE,RIMAG)                                           
   55 RE = DREAL(Q)                                                     
      RIMAG = DIMAG(Q)                                                  
   60 W = DCMPLX(RE,RIMAG)                                              
      GO TO 9005                                                        
 9000 CALL UERTST(IER,6HMERRCZ)                                         
 9005 RETURN                                                            
      END                                                               

C   IMSL ROUTINE NAME   - UERTST                                        
C                                                                       
C-----------------------------------------------------------------------
C                                                                       
C   COMPUTER            - IBM/SINGLE                                    
C                                                                       
C   LATEST REVISION     - JUNE 1, 1982                                  
C                                                                       
C   PURPOSE             - PRINT A MESSAGE REFLECTING AN ERROR CONDITION 
C                                                                       
C   USAGE               - CALL UERTST (IER,NAME)                        
C                                                                       
C   ARGUMENTS    IER    - ERROR PARAMETER. (INPUT)                      
C                           IER = I+J WHERE                             
C                             I = 128 IMPLIES TERMINAL ERROR MESSAGE,   
C                             I =  64 IMPLIES WARNING WITH FIX MESSAGE, 
C                             I =  32 IMPLIES WARNING MESSAGE.          
C                             J = ERROR CODE RELEVANT TO CALLING        
C                                 ROUTINE.                              
C                NAME   - A CHARACTER STRING OF LENGTH SIX PROVIDING    
C                           THE NAME OF THE CALLING ROUTINE. (INPUT)    
C                                                                       
C   PRECISION/HARDWARE  - SINGLE/ALL                                    
C                                                                       
C   REQD. IMSL ROUTINES - UGETIO,USPKD                                  
C                                                                       
C   NOTATION            - INFORMATION ON SPECIAL NOTATION AND           
C                           CONVENTIONS IS AVAILABLE IN THE MANUAL      
C                           INTRODUCTION OR THROUGH IMSL ROUTINE UHELP  
C                                                                       
C   REMARKS      THE ERROR MESSAGE PRODUCED BY UERTST IS WRITTEN        
C                TO THE STANDARD OUTPUT UNIT. THE OUTPUT UNIT           
C                NUMBER CAN BE DETERMINED BY CALLING UGETIO AS          
C                FOLLOWS..   CALL UGETIO(1,NIN,NOUT).                   
C                THE OUTPUT UNIT NUMBER CAN BE CHANGED BY CALLING       
C                UGETIO AS FOLLOWS..                                    
C                                NIN = 0                                
C                                NOUT = NEW OUTPUT UNIT NUMBER          
C                                CALL UGETIO(3,NIN,NOUT)                
C                SEE THE UGETIO DOCUMENT FOR MORE DETAILS.              
C                                                                       
C   COPYRIGHT           - 1982 BY IMSL, INC. ALL RIGHTS RESERVED.       
C                                                                       
C   WARRANTY            - IMSL WARRANTS ONLY THAT IMSL TESTING HAS BEEN 
C                           APPLIED TO THIS CODE. NO OTHER WARRANTY,    
C                           EXPRESSED OR IMPLIED, IS APPLICABLE.        
C                                                                       
C-----------------------------------------------------------------------
C                                                                       
      SUBROUTINE UERTST (IER,NAME)                                      
C                                  SPECIFICATIONS FOR ARGUMENTS         
      INTEGER            IER                                            
      INTEGER            NAME(1)                                        
C                                  SPECIFICATIONS FOR LOCAL VARIABLES   
      INTEGER            I,IEQ,IEQDF,IOUNIT,LEVEL,LEVOLD,NAMEQ(6),      
     *                   NAMSET(6),NAMUPK(6),NIN,NMTB                   
      DATA               NAMSET/1HU,1HE,1HR,1HS,1HE,1HT/                
      DATA               NAMEQ/6*1H /                                   
      DATA               LEVEL/4/,IEQDF/0/,IEQ/1H=/                     
C                                  UNPACK NAME INTO NAMUPK              
C                                  FIRST EXECUTABLE STATEMENT           
C-----------------------------------------------------------------------
CALL LIB MONITOR FROM UERTST, MAINTENANCE NUMBER 3034, DATE 82328       
C       CALL LIBMON( 3034 )                                             
C***PLEASE DON'T REMOVE OR CHANGE THE ABOVE CALL.  IT IS YOUR ONLY      
C***PROTECTION AGAINST YOUR USING AN OUT-OF-DATE OR INCORRECT           
C***VERSION OF THE ROUTINE.  THE LIBRARY MONITOR REMOVES THIS CALL,     
C***SO IT ONLY OCCURS ONCE, ON THE FIRST ENTRY TO THIS ROUTINE.         
C-----------------------------------------------------------------------
      CALL USPKD (NAME,6,NAMUPK,NMTB)                                   
C                                  GET OUTPUT UNIT NUMBER               
      CALL UGETIO(1,NIN,IOUNIT)                                         
C                                  CHECK IER                            
      IF (IER.GT.999) GO TO 25                                          
      IF (IER.LT.-32) GO TO 55                                          
      IF (IER.LE.128) GO TO 5                                           
      IF (LEVEL.LT.1) GO TO 30                                          
C                                  PRINT TERMINAL MESSAGE               
      IF (IEQDF.EQ.1) WRITE(*,35) IER,NAMEQ,IEQ,NAMUPK             
      IF (IEQDF.EQ.0) WRITE(*,35) IER,NAMUPK                       
      GO TO 30                                                          
    5 IF (IER.LE.64) GO TO 10                                           
      IF (LEVEL.LT.2) GO TO 30                                          
C                                  PRINT WARNING WITH FIX MESSAGE       
      IF (IEQDF.EQ.1) WRITE(*,40) IER,NAMEQ,IEQ,NAMUPK             
      IF (IEQDF.EQ.0) WRITE(*,40) IER,NAMUPK                       
      GO TO 30                                                          
   10 IF (IER.LE.32) GO TO 15                                           
C                                  PRINT WARNING MESSAGE                
      IF (LEVEL.LT.3) GO TO 30                                          
      IF (IEQDF.EQ.1) WRITE(*,45) IER,NAMEQ,IEQ,NAMUPK             
      IF (IEQDF.EQ.0) WRITE(*,45) IER,NAMUPK                       
      GO TO 30                                                          
   15 CONTINUE                                                          
C                                  CHECK FOR UERSET CALL                
      DO 20 I=1,6                                                       
         IF (NAMUPK(I).NE.NAMSET(I)) GO TO 25                           
   20 CONTINUE                                                          
      LEVOLD = LEVEL                                                    
      LEVEL = IER                                                       
      IER = LEVOLD                                                      
      IF (LEVEL.LT.0) LEVEL = 4                                         
      IF (LEVEL.GT.4) LEVEL = 4                                         
      GO TO 30                                                          
   25 CONTINUE                                                          
      IF (LEVEL.LT.4) GO TO 30                                          
C                                  PRINT NON-DEFINED MESSAGE            
      IF (IEQDF.EQ.1) WRITE(*,50) IER,NAMEQ,IEQ,NAMUPK             
      IF (IEQDF.EQ.0) WRITE(*,50) IER,NAMUPK                       
   30 IEQDF = 0                                                         
      RETURN                                                            
   35 FORMAT(19H *** TERMINAL ERROR,10X,7H(IER = ,I3,                   
     1       20H) FROM IMSL ROUTINE ,6A1,A1,6A1)                        
   40 FORMAT(27H *** WARNING WITH FIX ERROR,2X,7H(IER = ,I3,            
     1       20H) FROM IMSL ROUTINE ,6A1,A1,6A1)                        
   45 FORMAT(18H *** WARNING ERROR,11X,7H(IER = ,I3,                    
     1       20H) FROM IMSL ROUTINE ,6A1,A1,6A1)                        
   50 FORMAT(20H *** UNDEFINED ERROR,9X,7H(IER = ,I5,                   
     1       20H) FROM IMSL ROUTINE ,6A1,A1,6A1)                        
C                                                                       
C                                  SAVE P FOR P = R CASE                
C                                    P IS THE PAGE NAMUPK               
C                                    R IS THE ROUTINE NAMUPK            
   55 IEQDF = 1                                                         
      DO 60 I=1,6                                                       
   60 NAMEQ(I) = NAMUPK(I)                                              
   65 RETURN                                                            
      END                                                               

C   IMSL ROUTINE NAME   - UGETIO                                        
C                                                                       
C-----------------------------------------------------------------------
C                                                                       
C   COMPUTER            - IBM/SINGLE                                    
C                                                                       
C   LATEST REVISION     - JUNE 1, 1981                                  
C                                                                       
C   PURPOSE             - TO RETRIEVE CURRENT VALUES AND TO SET NEW     
C                           VALUES FOR INPUT AND OUTPUT UNIT            
C                           IDENTIFIERS.                                
C                                                                       
C   USAGE               - CALL UGETIO(IOPT,NIN,NOUT)                    
C                                                                       
C   ARGUMENTS    IOPT   - OPTION PARAMETER. (INPUT)                     
C                           IF IOPT=1, THE CURRENT INPUT AND OUTPUT     
C                           UNIT IDENTIFIER VALUES ARE RETURNED IN NIN  
C                           AND NOUT, RESPECTIVELY.                     
C                           IF IOPT=2, THE INTERNAL VALUE OF NIN IS     
C                           RESET FOR SUBSEQUENT USE.                   
C                           IF IOPT=3, THE INTERNAL VALUE OF NOUT IS    
C                           RESET FOR SUBSEQUENT USE.                   
C                NIN    - INPUT UNIT IDENTIFIER.                        
C                           OUTPUT IF IOPT=1, INPUT IF IOPT=2.          
C                NOUT   - OUTPUT UNIT IDENTIFIER.                       
C                           OUTPUT IF IOPT=1, INPUT IF IOPT=3.          
C                                                                       
C   PRECISION/HARDWARE  - SINGLE/ALL                                    
C                                                                       
C   REQD. IMSL ROUTINES - NONE REQUIRED                                 
C                                                                       
C   NOTATION            - INFORMATION ON SPECIAL NOTATION AND           
C                           CONVENTIONS IS AVAILABLE IN THE MANUAL      
C                           INTRODUCTION OR THROUGH IMSL ROUTINE UHELP  
C                                                                       
C   REMARKS      EACH IMSL ROUTINE THAT PERFORMS INPUT AND/OR OUTPUT    
C                OPERATIONS CALLS UGETIO TO OBTAIN THE CURRENT UNIT     
C                IDENTIFIER VALUES. IF UGETIO IS CALLED WITH IOPT=2 OR  
C                IOPT=3, NEW UNIT IDENTIFIER VALUES ARE ESTABLISHED.    
C                SUBSEQUENT INPUT/OUTPUT IS PERFORMED ON THE NEW UNITS. 
C                                                                       
C   COPYRIGHT           - 1978 BY IMSL, INC. ALL RIGHTS RESERVED.       
C                                                                       
C   WARRANTY            - IMSL WARRANTS ONLY THAT IMSL TESTING HAS BEEN 
C                           APPLIED TO THIS CODE. NO OTHER WARRANTY,    
C                           EXPRESSED OR IMPLIED, IS APPLICABLE.        
C                                                                       
C-----------------------------------------------------------------------
C                                                                       
      SUBROUTINE UGETIO(IOPT,NIN,NOUT)                                  
C                                  SPECIFICATIONS FOR ARGUMENTS         
      INTEGER            IOPT,NIN,NOUT                                  
C                                  SPECIFICATIONS FOR LOCAL VARIABLES   
      INTEGER            NIND,NOUTD                                     
      DATA               NIND/5/,NOUTD/6/                               
C                                  FIRST EXECUTABLE STATEMENT           
C-----------------------------------------------------------------------
CALL LIB MONITOR FROM UGETIO, MAINTENANCE NUMBER 3035, DATE 82328       
C       CALL LIBMON( 3035 )                                             
C***PLEASE DON'T REMOVE OR CHANGE THE ABOVE CALL.  IT IS YOUR ONLY      
C***PROTECTION AGAINST YOUR USING AN OUT-OF-DATE OR INCORRECT           
C***VERSION OF THE ROUTINE.  THE LIBRARY MONITOR REMOVES THIS CALL,     
C***SO IT ONLY OCCURS ONCE, ON THE FIRST ENTRY TO THIS ROUTINE.         
C-----------------------------------------------------------------------
      IF (IOPT.EQ.3) GO TO 10                                           
      IF (IOPT.EQ.2) GO TO 5                                            
      IF (IOPT.NE.1) GO TO 9005                                         
      NIN = NIND                                                        
      NOUT = NOUTD                                                      
      GO TO 9005                                                        
    5 NIND = NIN                                                        
      GO TO 9005                                                        
   10 NOUTD = NOUT                                                      
 9005 RETURN                                                            
      END                                                               


C   IMSL ROUTINE NAME   - USPKD                                         
C                                                                       
C-----------------------------------------------------------------------
C                                                                       
C   COMPUTER            - IBM/SINGLE                                    
C                                                                       
C   LATEST REVISION     - NOVEMBER 1, 1984                              
C                                                                       
C   PURPOSE             - NUCLEUS CALLED BY IMSL ROUTINES THAT HAVE     
C                           CHARACTER STRING ARGUMENTS                  
C                                                                       
C   USAGE               - CALL USPKD  (PACKED,NCHARS,UNPAKD,NCHMTB)     
C                                                                       
C   ARGUMENTS    PACKED - CHARACTER STRING TO BE UNPACKED.(INPUT)       
C                NCHARS - LENGTH OF PACKED. (INPUT)  SEE REMARKS.       
C                UNPAKD - INTEGER ARRAY TO RECEIVE THE UNPACKED         
C                         REPRESENTATION OF THE STRING. (OUTPUT)        
C                NCHMTB - NCHARS MINUS TRAILING BLANKS. (OUTPUT)        
C                                                                       
C   PRECISION/HARDWARE  - SINGLE/ALL                                    
C                                                                       
C   REQD. IMSL ROUTINES - NONE                                          
C                                                                       
C   REMARKS  1.  USPKD UNPACKS A CHARACTER STRING INTO AN INTEGER ARRAY 
C                IN (A1) FORMAT.                                        
C            2.  UP TO 129 CHARACTERS MAY BE USED.  ANY IN EXCESS OF    
C                THAT ARE IGNORED.                                      
C                                                                       
C   COPYRIGHT           - 1984 BY IMSL, INC.  ALL RIGHTS RESERVED.      
C                                                                       
C   WARRANTY            - IMSL WARRANTS ONLY THAT IMSL TESTING HAS BEEN 
C                           APPLIED TO THIS CODE.  NO OTHER WARRANTY,   
C                           EXPRESSED OR IMPLIED, IS APPLICABLE.        
C                                                                       
C-----------------------------------------------------------------------
      SUBROUTINE USPKD  (PACKED,NCHARS,UNPAKD,NCHMTB)                   
C                                  SPECIFICATIONS FOR ARGUMENTS         
      INTEGER            NC,NCHARS,NCHMTB                               
C                                                                       
      LOGICAL*1          UNPAKD(1),PACKED(1),LBYTE,LBLANK               
      INTEGER*2          IBYTE,IBLANK                                   
      EQUIVALENCE (LBYTE,IBYTE)                                         
      DATA               LBLANK /1H /                                   
      DATA               IBYTE /1H /                                    
      DATA               IBLANK /1H /                                   
C                                  INITIALIZE NCHMTB                    
C-----------------------------------------------------------------------
CALL LIB MONITOR FROM USPKD, MAINTENANCE NUMBER 3050, DATE 82328        
C        CALL LIBMON( 3050 )                                             
C***PLEASE DON'T REMOVE OR CHANGE THE ABOVE CALL.  IT IS YOUR ONLY      
C***PROTECTION AGAINST YOUR USING AN OUT-OF-DATE OR INCORRECT           
C***VERSION OF THE ROUTINE.  THE LIBRARY MONITOR REMOVES THIS CALL,     
C***SO IT ONLY OCCURS ONCE, ON THE FIRST ENTRY TO THIS ROUTINE.         
C-----------------------------------------------------------------------
      NCHMTB = 0                                                        
C                                  RETURN IF NCHARS IS LE ZERO          
      IF(NCHARS.LE.0) RETURN                                            
C                                  SET NC=NUMBER OF CHARS TO BE DECODED 
      NC = MIN0 (129,NCHARS)                                            
      NWORDS = NC*4                                                     
      J = 1                                                             
      DO 110 I = 1,NWORDS,4                                             
      UNPAKD(I) = PACKED(J)                                             
      UNPAKD(I+1) = LBLANK                                              
      UNPAKD(I+2) = LBLANK                                              
      UNPAKD(I+3) = LBLANK                                              
  110 J = J+1                                                           
C                                  CHECK UNPAKD ARRAY AND SET NCHMTB    
C                                  BASED ON TRAILING BLANKS FOUND       
      DO 200 N = 1,NWORDS,4                                             
         NN = NWORDS - N - 2                                            
         LBYTE = UNPAKD(NN)                                             
         IF(IBYTE .NE. IBLANK) GO TO 210                                
  200 CONTINUE                                                          
      NN = 0                                                            
  210 NCHMTB = (NN + 3) / 4                                             
      RETURN                                                            
      END                                                               
