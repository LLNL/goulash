# Goulash Generator for all rush_larsen tests

The generate_source script takes three compiler makedef files and generates all the rush larsen test directories and files using the templates in the Generator directory.   

The Goulash github repo was populated with:  
./generate_source cce-12.0.1-makedefs rocmcc-4.2.0-makedefs gcc-10.2.1-makedefs  

Running ./generate_source with no arguments shows usage info:  

Generates all rush_larsen tests include up to 3 compiler interop tests  
plus generates make files using compiler setting in the passed make-def files  
 
Usage: ./generate_source compiler1-makedefs compiler2-makedefs compiler3-makedefs  
   
  Goulash github rush_larsen tests created with:  
    ./generate_source cce-12.0.1-makedefs rocmcc-4.2.0-makedefs gcc-10.2.1-makedefs  
   
  Start testing with just one compiler by repeating same defs three times:  
    ./generate_source cce-12.0.1-makedefs cce-12.0.1-makedefs cce-12.0.1-makedefs  
    ./generate_source rocmcc-4.2.0-makedefs rocmcc-4.2.0-makedefs rocmcc-4.2.0-makedefs  
   
Recommend starting with individual tests first before doing interop versions  
   
NOTE: In each directory, 'make' builds test and 'make check' runs test  
   
WARNING: Overwrites all Makefiles and source files in ../rush_larsen*!  
         You may want to run rm -rf ../rush_larsen_* before this command    
