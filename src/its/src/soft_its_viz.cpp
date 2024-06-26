#include <GL/glut.h>
#include <math.h>
#include <cstdio>
#include "its_msgs/SoftContactSensingProblemSolution.h"
#include <ros/ros.h>

//!!Define all the parameters
double A , B, C;
double force_th;

// Drawing params
enum class Theme: unsigned short int {
    Dark = 1,    
    Light = 2,
};
//Theme theme = Theme::Light;
Theme theme = Theme::Dark;
int changingcolor=0;
int ch=0;
bool fullscreen = false;
bool mouseDown = false;

float xrot = 0.0f;
float yrot = 0.0f;

float xdiff = 0.0f;
float ydiff = 0.0f;

// PoC view axis
float fx=0;
float fy=0;
float fz=-3;

// SITS Solution
float x;
float y;
float z;
float d;

// Force and Torque
float fn[3];
float ft[3];
float lt;
//!!End of parameter definatio

void drawAxis(){
	int len = 10;

	glBegin(GL_LINES);	
	// x-axis
	glColor3f(1,0,0);
	glVertex3f(.0f, .0f, .0f);
	glVertex3f(len, .0f, .0f);
	// y-axis
	glColor3f(0,1,0);
	glVertex3f(.0f, .0f, .0f);
	glVertex3f(.0f, len, .0f);
	// z-axis
	glColor3f(0,0,1);
	glVertex3f(.0f,.0f, .0f);
	glVertex3f(.0f, .0f, len);

	glEnd();


}

void drawAxis_v2(){
	int len = 1;
	int dispX = -2;
	int dispY = -1.5;
	
	// rotate as view
	glRotated(-yrot, 0.0f, 1.0f, 0.0f);
	glRotated(-xrot, 1.0f, 0.0f, 0.0f);
	// move to bootom left
	glTranslated(dispX,dispY,0);
	// restore as surfaces
	glRotated(xrot, 1.0f, 0.0f, 0.0f);
	glRotated(yrot, 0.0f, 1.0f, 0.0f);

	glBegin(GL_LINES);	
	// x-axis
	glColor3f(1,0,0);
	glVertex3f(.0f, .0f, .0f);
	glVertex3f(len, .0f, .0f);
	// y-axis
	glColor3f(0,1,0);
	glVertex3f(.0f, .0f, .0f);
	glVertex3f(.0f, len, .0f);
	// z-axis
	glColor3f(0,0,1);
	glVertex3f(.0f,.0f, .0f);
	glVertex3f(.0f, .0f, len);
	glEnd();
	
	// rotate as view
	glRotated(-yrot, 0.0f, 1.0f, 0.0f);
	glRotated(-xrot, 1.0f, 0.0f, 0.0f);
	// return to center
	glTranslated(-dispX,-dispY,0);
	// rotate as surfaces
	glRotated(xrot, 1.0f, 0.0f, 0.0f);
	glRotated(yrot, 0.0f, 1.0f, 0.0f);



}
void drawTorque(float lt)
{
	float fRadius = 0.2f;
	//float fRadius = 0.4f;
	float fPrecision = 0.05f;
	float fCenterX = 0.0f;
	float fCenterY = 0.0f;
	float fAngle;
	float fX=0.00f;
	float fY=0.0f;

	glBegin(GL_LINE_STRIP);
	for(fAngle = 0.0f; fAngle <= fabs(0.05*lt * 3.14159); fAngle += fPrecision)
	{
		fX = fCenterX + (fRadius* static_cast<float>(sin(fAngle)));
		fY = fCenterY + (fRadius* static_cast<float>(cos(fAngle)));
		if (lt>0)  fX=-fX;
		glVertex3f(fX, fY, 0);
	}
	glEnd();

	glBegin(GL_TRIANGLES);						// Drawing Using Triangles
	glVertex3f( fX,fY+0.06f,0);					// Top //Default all 0.02
	glVertex3f( fX,fY-0.06f,0);					// Bottom Left
	glVertex3f( fX+0.06f,fY,0);					// Bottom Right
	glEnd();									// Finished Drawing The Triangle
}

void drawBox()
{
	glBegin(GL_QUADS);
	glColor3f(1.0f, 0.0f, 0.0f);
	// BOTTOM
	glVertex3f(-0.5f, -1.5f, 0.5f);
	glVertex3f(-0.5f, -1.5f, -0.5f);
	glVertex3f( 0.5f, -1.5f, -0.5f);
	glVertex3f( 0.5f, -1.5f, 0.5f);
	glEnd();
}

void light()
{
	glEnable(GL_LIGHTING);
	GLfloat specular[] = {1.0, 1.0, 1.0, 1.0};
	GLfloat ambientLight[] = { 0.2f, 0.2f, 0.2f, 1.0f };

	glLightfv(GL_LIGHT0, GL_SPECULAR, specular);
	glShadeModel(GL_SMOOTH);
	glEnable(GL_LIGHT0);
	glLightfv(GL_LIGHT0, GL_AMBIENT, ambientLight);
	GLfloat position[] = { 0, 0, 5.0f, 1.0f };
	glLightfv(GL_LIGHT0, GL_POSITION, position);

	position[2] = 5.0;
	glLightfv(GL_LIGHT1, GL_POSITION, position);
	glLightfv(GL_LIGHT1, GL_SPECULAR, specular);
	glEnable(GL_LIGHT1);
}

void drawEllipsoid(float a, float b, float c, int lats, int longs, bool transparency = false)
{
	float mcolor[4];
	if (transparency){
		mcolor[0] = 0.8; mcolor[1] = 0.8; mcolor[2] = 0.8; mcolor[3] = .1;
	}else{
		mcolor[0] = 0.8; mcolor[1] = 0.8; mcolor[2] = 0.8; mcolor[3] = 1;
	}
	glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, mcolor);

	int i, j;
	for(i = lats/2; i <= lats; i++)
	{
		float lat0 =  M_PI * (-0.5f + (float) (i - 1) / lats);
		float z0 = sin(lat0);
		float zr0 = cos(lat0);

		float lat1 = M_PI * (-0.5f + (float) i / lats);
		float z1 = sin(lat1);
		float zr1 = cos(lat1);

		glBegin(GL_QUAD_STRIP);
		for(j = 0; j <= longs; j++)
		{
			float lng = 2* M_PI * (float) (j - 1) / longs;
			float x = cos(lng);
			float y = sin(lng);

			glNormal3f(x * zr0, y * zr0, z0);
			glVertex3f(x * zr0 * a, y * zr0 * b, z0 * c);
			glNormal3f(x * zr1, y * zr1, z1);
			glVertex3f(x * zr1 * a, y * zr1 * b, z1 * c);
		}
		glEnd();
	}
}

bool init(){
	switch (theme){
		case Theme::Dark:
			glClearColor(0.15f, 0.15f, 0.15f, 0.0f);
			break;
		case Theme::Light:
			glClearColor(0.93f, 0.93f, 0.93f, 0.0f);
			break;	
		default:
			glClearColor(0.93f, 0.93f, 0.93f, 0.0f); 	// Default: Light theme
			break;
	}
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glClearDepth(1.0f);

	return true;
}

void display(){
	// restore view 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();
	gluLookAt(
			0.0f, 0.0f, 6.0f,	// eye position (x, y, z)
			0.0f, 0.0f, 0.0f,	// reference position (x, y, z)
			0.0f, 1.0f, 0.0f);	// elevation (x, y, z)
	glRotatef(xrot, 1.0f, 0.0f, 0.0f);
	glRotatef(yrot, 0.0f, 1.0f, 0.0f);


	// set depth properties
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);


	// draw Axis
	//drawAxis();
	drawAxis_v2();

	// light up
	light();

	//	Fingertip Surface
	drawEllipsoid(A-d,B-d,C-d, 15,30);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); 
	drawEllipsoid(A,B,C, 15,30, true);
	glDisable(GL_BLEND);
	glDisable(GL_DEPTH_TEST);



	// flicking edge color
	if(changingcolor==100) changingcolor=0;
	changingcolor++;
	float ecolor[] = { static_cast<float>(0.1*changingcolor/10.0), 0.0f, static_cast<float>(1-0.1*changingcolor/10), 0.8f };
	glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, ecolor);

	glTranslated(x,y,z);	// center drawings in PoC
	
	// Normal Force
	glBegin(GL_LINES);
	glVertex3f(0,0,0);
	glVertex3f(fn[0]/10,fn[1]/10, fn[2]/10);		
	glEnd( );

	// Tangential Force
	glBegin(GL_LINES);
	glVertex3f(0,0,0);
	glVertex3f(ft[0]/10,ft[1]/10, ft[2]/10);
	glEnd( );


	// Total Force components in x, y, z direction
	double Ftot=sqrt((fn[0]+ft[0])*(fn[0]+ft[0])+(fn[1]+ft[1])*(fn[1]+ft[1])+(fn[2]+ft[2])*(fn[2]+ft[2]));
	double fx=fn[0]+ft[0];
	double fy=fn[1]+ft[1];
	double fz=fn[2]+ft[2];
	
	// rotate to PoC view to align w.r.t Force axis
	float theta1=180-(180/M_PI)*atan2(fz,fx);
	float theta2=(180/M_PI)*atan2(fy, sqrt(fx*fx+fz*fz));
	glRotated( 90, 0, 1,0);
	glRotated(theta1, 0, 1, 0);
	glRotated(theta2 ,1, 0, 0);

	if(Ftot > force_th){
		// Draw in PoC
		static GLUquadricObj *q;
		q = gluNewQuadric();
		gluQuadricNormals(q,GLU_SMOOTH);
		gluCylinder(q,0.01,0.05*Ftot,Ftot/3,30,20);
		
		// Draw Torque
		drawTorque(lt);
	}

	glDisable(GL_LIGHTING);
	glFlush();
	glutSwapBuffers();
	glutPostRedisplay();
}

void resize(int w, int h)
{
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glViewport(0, 0, w, h);
	gluPerspective(45.0f, 1.0f * w / h, 1.0f, 100.0f);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}

void idle()
{
/**	if (!mouseDown)
	{
		xrot += 0.3f;
		yrot += 0.4f;
	}*/		//Use when necessary
	ros::spinOnce();
	usleep(10000);
	//glutPostRedisplay();
}

void keyboard(unsigned char key, int x, int y) 
{
	switch(key)
	{
		case 27 :
			exit(1); 
			break;
	}
}

void specialKeyboard(int key, int x, int y)
{
	if (key == GLUT_KEY_F1)
	{
		fullscreen = !fullscreen;
		if (fullscreen)
			glutFullScreen();
		else
		{
			glutReshapeWindow(500, 500);
			glutPositionWindow(50, 50);
		}
	}
}

void mouse(int button, int state, int x, int y)
{
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN)
	{
		mouseDown = true;
		xdiff = x - yrot;
		ydiff = -y + xrot;
	}
	else
		mouseDown = false;
}

void mouseMotion(int x, int y)
{
	if (mouseDown)
	{
		yrot = x - xdiff;
		xrot = y + ydiff;
		glutPostRedisplay();
	}
}

void Callback_contactstate(const its_msgs::SoftContactSensingProblemSolution::ConstPtr& msg)
{
	//	Contact location [mm] (x, y, z)
	x=msg->PoC.x/10;
	z=msg->PoC.z/10;
	y=msg->PoC.y/10;
	//	Deformation [mm] Dd
    d = msg->D/10;
	//	Normal force Fnormal [N]
	double Fnormal=msg->Fn;
	fn[0]=msg->n.x*Fnormal;
	fn[1]=msg->n.y*Fnormal;
	fn[2]=msg->n.z*Fnormal;
	//	Tangential force [N]
	ft[0]=msg->Ft.x;
	ft[1]=msg->Ft.y;
	ft[2]=msg->Ft.z;
	//	Force components in x, y, z direction
	//	Local torque [Nmm]
	lt=(float) msg->T;

	glutPostRedisplay();
}

int main(int argc, char *argv[])
{
	ros::init(argc, argv, "soft_csp_viz");
	ros::NodeHandle nh;
	ros::Subscriber sub = nh.subscribe("soft_csp/solution", 1000, Callback_contactstate);

	//	Declaring parameters
    nh.param<double>("fingertip/principalSemiAxis/a",A, 15);
    nh.param<double>("fingertip/principalSemiAxis/b",B, 15);
    nh.param<double>("fingertip/principalSemiAxis/c",C, 15);  
    A/=10;B/=10;C/=10;

    nh.param<double>("algorithm/force_threshold",force_th, 0.5);  

	std::string theme_str;
    nh.param<std::string>("soft_viz/theme",theme_str, "Dark");  
	if(theme_str=="Dark"){
		theme = Theme::Dark;
	}else{
		theme = Theme::Light;
	}

	glutInit(&argc, argv);
	glutInitWindowPosition(50, 50);
	glutInitWindowSize(500, 500);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
	std::string wintitle="SoftCSPViz ";
	glutCreateWindow(wintitle.c_str());

	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutSpecialFunc(specialKeyboard);
	glutMouseFunc(mouse);
	glutMotionFunc(mouseMotion);
	glutReshapeFunc(resize);
	glutIdleFunc(idle);
	if (!init())
		return 1;
	glutMainLoop();
	return 0;
}